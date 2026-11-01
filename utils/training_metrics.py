"""Task training metrics, shared verbatim by the five comparison frameworks.

See TRAINING_METRICS.md at the workspace root for the measurement contract.
No extra forward/backward pass, tensor copy, or retained autograd graph is used.
"""

import json
import logging
import os
from collections import Counter
from contextlib import contextmanager

import numpy as np
import torch

try:
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import flop_registry
except ImportError:
    TorchDispatchMode = object
    flop_registry = None


CONTRACT = "task-training-v1-core-flops"
_SKIP_STATE = {
    "args", "config", "tuning_config", "data_manager", "_data_manager",
    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
    "_state_dict_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
}


class _FlopCounter(TorchDispatchMode):
    """Count executed ATen operations, including backward, without storing events."""

    def __init__(self):
        super().__init__()
        self.registry = dict(flop_registry)
        # PyTorch 2.6's registry omits CPU flash attention. Use the same formula.
        for suffix in ("", "_backward"):
            cpu = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu" + suffix, None)
            gpu = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention" + suffix, None)
            if cpu is not None and gpu in self.registry:
                self.registry.setdefault(cpu, self.registry[gpu])
        self.flops = Counter()
        self.uncovered = Counter()
        self.errors = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        packet = func._overloadpacket
        formula = self.registry.get(packet)
        if formula is None:
            self.uncovered[str(packet)] += 1
        else:
            try:
                self.flops[str(packet)] += int(formula(*args, **kwargs, out_val=output))
            except Exception as exc:
                # A metric failure must not repeat the operation or alter training.
                self.errors.add("{}: {}".format(packet, exc))
        return output


class _StorageCounter:
    """Unique retained tensor/array backing stores, plus replay file payloads."""

    def __init__(self):
        self.seen_objects = set()
        self.seen_storages = set()
        self.seen_files = set()
        self.bytes = Counter()
        self.errors = []

    def visit(self, value, category):
        if id(value) in self.seen_objects:
            return
        self.seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type == "meta":
                self.errors.append("meta tensor has no materialized storage")
                return
            if value.layout != torch.strided:
                self.errors.append("unsupported tensor layout: {}".format(value.layout))
                return
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in self.seen_storages:
                self.seen_storages.add(key)
                self.bytes[category] += storage.nbytes()
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                for item in value.flat:
                    self.visit(item, category)
            else:
                root = value
                while isinstance(root.base, np.ndarray):
                    root = root.base
                # Also deduplicates a NumPy view of a CPU tensor.
                if isinstance(root.base, torch.Tensor):
                    self.visit(root.base, category)
                else:
                    key = ("cpu", root.__array_interface__["data"][0], root.nbytes)
                    if key not in self.seen_storages:
                        self.seen_storages.add(key)
                        self.bytes[category] += root.nbytes
        elif isinstance(value, torch.nn.Module):
            for name, item in vars(value).items():
                if name not in _SKIP_STATE:
                    self.visit(item, category)
        elif isinstance(value, dict):
            for name, item in value.items():
                if not isinstance(name, str) or name not in _SKIP_STATE:
                    self.visit(item, category)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                self.visit(item, category)
        # DataLoader, Dataset, optimizer, scheduler, Python metadata and callables
        # are deliberately not recursively traversed.

    def replay_files(self, value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(value))
            if path not in self.seen_files:
                self.seen_files.add(path)
                try:
                    self.bytes["replay_files"] += os.path.getsize(path)
                except OSError as exc:
                    self.errors.append("replay file {}: {}".format(path, exc))
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in "USO":
                for item in value.flat:
                    self.replay_files(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.replay_files(item)


def persistent_storage(learner):
    counter = _StorageCounter()
    state = vars(learner)
    # Count modules first so aliases to model weights are assigned consistently.
    for value in state.values():
        if isinstance(value, torch.nn.Module):
            counter.visit(value, "model_state")
    for name in ("_data_memory", "_targets_memory"):
        counter.visit(state.get(name), "replay_state")
    counter.replay_files(state.get("_data_memory"))
    for name, value in state.items():
        if name not in _SKIP_STATE:
            counter.visit(value, "auxiliary_state")
    parts = {name: counter.bytes[name] for name in
             ("model_state", "auxiliary_state", "replay_state", "replay_files")}
    return {
        "persistent_storage_bytes": None if counter.errors else sum(parts.values()),
        "persistent_storage_breakdown_bytes": parts,
        "storage_errors": counter.errors,
    }


def _cuda_devices(args):
    if not torch.cuda.is_available():
        return []
    devices = []
    for value in args.get("device", []):
        device = torch.device("cpu" if value == -1 else
                              "cuda:{}".format(value) if isinstance(value, int) else value)
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index not in devices:
                devices.append(index)
    return devices


class TrainingMetrics:
    """One tracker per seed; call measure() around training and task finalization."""

    def __init__(self, args):
        self.enabled = args.get("profile_training", True)
        self.devices = _cuda_devices(args) if self.enabled else []
        self.cumulative_flops = 0
        self.task = None
        self.phases = []
        self.flop_status = "ok"
        if flop_registry is None:
            self.flop_status = "unavailable: torch.utils.flop_counter is required"
        elif len(args.get("device", [])) > 1:
            # Dispatch modes are thread-local: DataParallel worker operations
            # would otherwise silently disappear from the count.
            self.flop_status = "unavailable: use one device for comparable FLOPs (DataParallel threads)"
        if self.enabled:
            logging.info(
                "Training metrics contract => %s; torch=%s; MAC=2 FLOPs; "
                "scope=incremental_train+after_task; storage=retained tensor/array/replay payload; "
                "FLOPs=executed registered matmul/convolution/attention ops, including backward; "
                "elementwise/reductions/optimizer scalar ops/NumPy linalg excluded; status=%s",
                CONTRACT, torch.__version__, self.flop_status,
            )

    @contextmanager
    def measure(self, task, phase):
        if not self.enabled:
            yield
            return
        if self.task != task:
            self.task = task
            self.phases = []
        for device in self.devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        counter = _FlopCounter() if self.flop_status == "ok" else None
        if counter is not None:
            counter.__enter__()
        try:
            yield
        finally:
            if counter is not None:
                counter.__exit__(None, None, None)
            peaks = {}
            for device in self.devices:
                torch.cuda.synchronize(device)
                peaks[str(device)] = torch.cuda.max_memory_allocated(device)
            self.phases.append({
                "phase": phase,
                "peak_allocated_bytes_by_device": peaks,
                "flops": sum(counter.flops.values()) if counter is not None and not counter.errors else None,
                "flops_by_operator": dict(counter.flops) if counter is not None else {},
                "uncovered_operator_calls": dict(counter.uncovered) if counter is not None else {},
                "flop_errors": sorted(counter.errors) if counter is not None else [],
            })

    def log_task(self, learner, task):
        if not self.enabled:
            return
        if self.task != task or [p["phase"] for p in self.phases] != ["incremental_train", "after_task"]:
            raise RuntimeError("Training metrics require incremental_train and after_task for this task")
        task_flops = None if any(p["flops"] is None for p in self.phases) else sum(p["flops"] for p in self.phases)
        if task_flops is None or self.cumulative_flops is None:
            self.cumulative_flops = None
        else:
            self.cumulative_flops += task_flops
        peaks = {str(d): max(p["peak_allocated_bytes_by_device"][str(d)] for p in self.phases)
                 for d in self.devices}
        try:
            storage = persistent_storage(learner)
        except Exception as exc:
            storage = {"persistent_storage_bytes": None, "storage_errors": [str(exc)]}
        result = {
            "contract": CONTRACT,
            "torch_version": torch.__version__,
            "task": task,
            "tasks_seen": task + 1,
            "peak_training_memory_bytes": max(peaks.values()) if peaks else None,
            "peak_allocated_bytes_by_device": peaks,
            "training_gflops": task_flops / 1e9 if task_flops is not None else None,
            "cumulative_training_gflops": self.cumulative_flops / 1e9 if self.cumulative_flops is not None else None,
            "flop_status": self.flop_status if task_flops is not None or self.flop_status != "ok" else "formula_error",
            "phases": self.phases,
            **storage,
        }
        def number(value, divisor=1):
            return "N/A" if value is None else "{:.6f}".format(value / divisor)

        logging.info(
            "Task %s Training Profile => Peak training memory: %s MiB, "
            "Total persistent storage: %s MiB, Training FLOPs/task: %s GFLOPs, "
            "Cumulative training FLOPs: %s GFLOPs",
            task, number(result["peak_training_memory_bytes"], 2 ** 20),
            number(result["persistent_storage_bytes"], 2 ** 20),
            number(result["training_gflops"]), number(result["cumulative_training_gflops"]),
        )
        logging.info("Task %s Training Profile JSON => %s", task, json.dumps(result, sort_keys=True))
        return result
