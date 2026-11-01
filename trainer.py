import sys
import logging
import copy
import torch
from utils import factory
from utils.training_metrics import TrainingMetrics
from utils.data_manager import DataManager
from utils.toolkit import (
    count_parameters,
    log_single_image_profile,
    profile_single_image_inference,
)
import os
from datetime import datetime
import time
import numpy as np
np.set_printoptions(linewidth=np.inf)



def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    clock0 = time.time()
    s = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    if 'in21k' in args["backbone_type"]:
        mn = '21K'
    else:
        mn = '1K'
    logs_name = "logs/{}/{}/{}/{}/{}".format(args["model_name"], args["dataset"], mn, init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}/{}".format(
        args["model_name"],
        args["dataset"],
        mn,
        init_cls,
        args["increment"],
        s,
    )
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="[%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    model_path = "save_models/{}/{}".format(args["model_name"], args["dataset"])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)
    training_metrics = TrainingMetrics(args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        with training_metrics.measure(task, "incremental_train"):
            model.incremental_train(data_manager)
        # Count after incremental_train has created this task's classifier and
        # proxy head, so the numbers describe the model that was just trained.
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        try:
            profile = profile_single_image_inference(
                model._network,
                task,
                input_size=(3, getattr(model, "size", 224), getattr(model, "size", 224)),
                device=args["device"][0],
                warmup=args.get("profile_warmup", 5),
                repeat=args.get("profile_repeat", 20),
            )
            log_single_image_profile(logging, profile)
        except Exception as exc:
            logging.info("Task {} Profile => failed: {}".format(task, exc))
        # model.save_checkpoint(model_path)
        # checkpoint = torch.load(f"{model_path}/{task}.pkl", weights_oly=False)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model = torch.load()
        cnn_accy, nme_accy = model.eval_task()
        # cnn_accy, nme_accy = model.eval_task_new()
        with training_metrics.measure(task, "after_task"):
            model.after_task()
        training_metrics.log_task(model, task)

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])


            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            # print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        BWT = np.mean(np_acctable[:-1, -1] - np.diag(np_acctable)[:-1])
        logging.info('Accuracy Matrix (CNN):\n{}'.format(np_acctable))
        logging.info('Forgetting (CNN): {}'.format(forgetting))
        logging.info('BWT (CNN): {}'.format(BWT))
    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        BWT = np.mean(np_acctable[:-1, -1] - np.diag(np_acctable)[:-1])
        logging.info('Accuracy Matrix (NME):\n{}'.format(np_acctable))
        logging.info('Forgetting (NME): {}'.format(forgetting))
        logging.info('BWT (NME): {}'.format(BWT))
    clock1 = time.time()
    logging.info("Total time: {:6.1f}s".format(clock1 - clock0))
    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
