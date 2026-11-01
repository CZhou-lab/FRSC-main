import os
import time
import numpy as np
import torch
from torch import nn


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _format_profile_number(value, unit):
    if value is None:
        return "N/A"
    if unit == "flops":
        return "{:.4f} GFLOPs".format(value / 1e9)
    if unit == "time":
        return "{:.4f} ms".format(value)
    return str(value)


def _count_inference_flops(model, inputs, forward_fn):
    flops = {"total": 0}
    handles = []

    def add_flops(value):
        flops["total"] += int(value)

    def linear_hook(module, module_inputs, output):
        if not module_inputs:
            return
        x = module_inputs[0]
        if not torch.is_tensor(x):
            return
        batch_ops = int(np.prod(x.shape[:-1])) if len(x.shape) > 1 else 1
        add_flops(batch_ops * module.in_features * module.out_features * 2)
        if module.bias is not None:
            add_flops(batch_ops * module.out_features)

    def conv_hook(module, module_inputs, output):
        if not module_inputs or not torch.is_tensor(output):
            return
        out = output
        out_elements = out.numel()
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
            * 2
        )
        add_flops(out_elements * kernel_ops)
        if module.bias is not None:
            add_flops(out_elements)

    def norm_hook(module, module_inputs, output):
        if torch.is_tensor(output):
            add_flops(output.numel() * 5)

    def activation_hook(module, module_inputs, output):
        if torch.is_tensor(output):
            add_flops(output.numel())

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            handles.append(module.register_forward_hook(norm_hook))
        elif isinstance(module, (nn.ReLU, nn.GELU)):
            handles.append(module.register_forward_hook(activation_hook))

    original_bmm = torch.bmm
    original_matmul = torch.matmul

    def counted_bmm(input, mat2, *args, **kwargs):
        result = original_bmm(input, mat2, *args, **kwargs)
        if input.dim() == 3 and mat2.dim() == 3:
            add_flops(input.shape[0] * input.shape[1] * input.shape[2] * mat2.shape[2] * 2)
        return result

    def counted_matmul(input, other, *args, **kwargs):
        result = original_matmul(input, other, *args, **kwargs)
        if torch.is_tensor(input) and torch.is_tensor(other) and input.dim() >= 2 and other.dim() >= 2:
            add_flops(int(np.prod(result.shape)) * input.shape[-1] * 2)
        return result

    try:
        torch.bmm = counted_bmm
        torch.matmul = counted_matmul
        with torch.no_grad():
            forward_fn(inputs)
    finally:
        torch.bmm = original_bmm
        torch.matmul = original_matmul
        for handle in handles:
            handle.remove()

    return flops["total"]


def _get_vit_adapter_flops(network, input_size):
    backbone = getattr(network, "backbone", None)
    if backbone is None or not all(hasattr(backbone, attr) for attr in ("patch_embed", "blocks", "embed_dim")):
        return None

    channels, height, width = input_size
    patch_embed = backbone.patch_embed
    proj = getattr(patch_embed, "proj", None)
    if proj is None or not hasattr(proj, "kernel_size"):
        return None

    embed_dim = backbone.embed_dim
    num_tokens = patch_embed.num_patches + getattr(backbone, "num_tokens", 1)
    depth = len(backbone.blocks)
    if depth == 0:
        return None

    patch_out_h = height // proj.stride[0]
    patch_out_w = width // proj.stride[1]
    patch_flops = (
        patch_out_h
        * patch_out_w
        * embed_dim
        * proj.kernel_size[0]
        * proj.kernel_size[1]
        * channels
        * 2
    )
    if proj.bias is not None:
        patch_flops += patch_out_h * patch_out_w * embed_dim

    sample_block = backbone.blocks[0]
    num_heads = sample_block.attn.num_heads
    mlp_hidden_dim = sample_block.fc1.out_features

    norm_flops = num_tokens * embed_dim * 5
    qkv_flops = 3 * num_tokens * embed_dim * embed_dim * 2
    qkv_bias_flops = 3 * num_tokens * embed_dim if sample_block.attn.q_proj.bias is not None else 0
    attn_flops = 2 * num_heads * num_tokens * num_tokens * (embed_dim // num_heads) * 2
    proj_flops = num_tokens * embed_dim * embed_dim * 2
    if sample_block.attn.proj.bias is not None:
        proj_flops += num_tokens * embed_dim
    mlp_flops = (
        num_tokens * embed_dim * mlp_hidden_dim * 2
        + num_tokens * mlp_hidden_dim
        + num_tokens * mlp_hidden_dim * embed_dim * 2
    )
    if sample_block.fc1.bias is not None:
        mlp_flops += num_tokens * mlp_hidden_dim
    if sample_block.fc2.bias is not None:
        mlp_flops += num_tokens * embed_dim
    block_flops = norm_flops + qkv_flops + qkv_bias_flops + attn_flops + proj_flops + norm_flops + mlp_flops

    adapter_count = len(getattr(backbone, "adapter_list", [])) + 1
    adapter_flops = 0
    if hasattr(backbone, "cur_adapter") and len(backbone.cur_adapter) > 0:
        sample_adapter = backbone.cur_adapter[0]
        bottleneck = sample_adapter.down_proj.out_features
        adapter_flops = (
            num_tokens * embed_dim * bottleneck * 2
            + num_tokens * bottleneck
            + num_tokens * bottleneck
            + num_tokens * bottleneck * embed_dim * 2
            + num_tokens * embed_dim
            + num_tokens * embed_dim
        )

    classifier_flops = 0
    fc = getattr(network, "fc", None)
    if fc is not None and getattr(fc, "weight", None) is not None:
        total_classes = fc.weight.shape[0]
        init_cls = getattr(network, "init_cls", total_classes)
        inc = getattr(network, "inc", 0)
        if adapter_count == 1:
            classifier_flops += (total_classes + 1) * embed_dim * 5
            classifier_flops += total_classes * embed_dim * 2
        else:
            classifier_flops += total_classes * embed_dim * 5
            classifier_flops += adapter_count * embed_dim * 5
            for task_idx in range(adapter_count):
                class_count = init_cls if task_idx == 0 else inc
                if task_idx > 0:
                    classifier_flops += (task_idx + 1) * embed_dim * 4
                    classifier_flops += task_idx + 1
                classifier_flops += embed_dim * 5
                classifier_flops += class_count * embed_dim * 2

    final_norm_flops = num_tokens * embed_dim * 5
    frozen_vit_flops = patch_flops + depth * block_flops + final_norm_flops
    single_image_flops = (
        patch_flops
        + adapter_count * (depth * (block_flops + adapter_flops) + final_norm_flops)
        + classifier_flops
    )

    return {
        "single_image_flops": single_image_flops,
        "frozen_vit_flops": frozen_vit_flops,
        "extra_flops": single_image_flops - frozen_vit_flops,
    }


def _measure_inference_time(inputs, forward_fn, warmup=5, repeat=20):
    device = inputs.device
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for _ in range(warmup):
            forward_fn(inputs)
        if use_cuda:
            torch.cuda.synchronize(device)

        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(repeat):
                forward_fn(inputs)
            end_event.record()
            torch.cuda.synchronize(device)
            return start_event.elapsed_time(end_event) / repeat

        start = time.perf_counter()
        for _ in range(repeat):
            forward_fn(inputs)
        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / repeat


def profile_single_image_inference(network, task_id, input_size=(3, 224, 224), device=None,
                                   warmup=5, repeat=20):
    was_training = network.training
    network.eval()

    if device is None:
        device = next(network.parameters()).device
    else:
        device = torch.device(device)

    inputs = torch.randn(1, *input_size, device=device)

    def adapter_forward(x):
        return network(x, test=True, task=task_id)

    backbone = getattr(network, "backbone", None)
    has_vit_forward = backbone is not None and hasattr(backbone, "forward_vit")

    def frozen_vit_forward(x):
        return backbone.forward_vit(x)

    result = {
        "task": task_id,
        "trainable_params": count_parameters(network, True),
        "single_image_flops": None,
        "frozen_vit_flops": None,
        "extra_flops": None,
        "single_image_time_ms": None,
        "input_size": input_size,
    }

    try:
        analytical_flops = _get_vit_adapter_flops(network, input_size)
        if analytical_flops is not None:
            result.update(analytical_flops)
        else:
            result["single_image_flops"] = _count_inference_flops(network, inputs, adapter_forward)
            if has_vit_forward:
                result["frozen_vit_flops"] = _count_inference_flops(network, inputs, frozen_vit_forward)
                result["extra_flops"] = result["single_image_flops"] - result["frozen_vit_flops"]
        result["single_image_time_ms"] = _measure_inference_time(inputs, adapter_forward, warmup=warmup, repeat=repeat)
    finally:
        if was_training:
            network.train()

    return result


def log_single_image_profile(logging_module, profile):
    logging_module.info(
        "Task {task} Profile => Trainable params: {params}, "
        "Inference FLOPs/image: {flops}, Frozen ViT FLOPs/image: {vit_flops}, "
        "Extra FLOPs/image: {extra_flops}, Inference time/image: {time_ms}, Input: {input_size}".format(
            task=profile["task"],
            params=profile["trainable_params"],
            flops=_format_profile_number(profile["single_image_flops"], "flops"),
            vit_flops=_format_profile_number(profile["frozen_vit_flops"], "flops"),
            extra_flops=_format_profile_number(profile["extra_flops"], "flops"),
            time_ms=_format_profile_number(profile["single_image_time_ms"], "time"),
            input_size=profile["input_size"],
        )
    )


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = float(np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    ))

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = float(np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        ))

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else float(np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        ))
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = float(np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    ))

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
