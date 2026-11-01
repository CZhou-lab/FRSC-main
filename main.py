import json
import argparse
from trainer import train

# import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def main():
    args = setup_parser().parse_args()
    cli_temp_project_before_mean = args.temp_project_before_mean
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    if cli_temp_project_before_mean:
        args["temp_project_before_mean"] = True

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Run FRSC for class-incremental learning.')
    parser.add_argument('--config', type=str, default='./exps/frsc_vtab_in21k.json',
                        help='Json file of settings.')
    parser.add_argument('--temp-project-before-mean', action='store_true',
                        help='Project each historical prototype with its adapter H @ H.T before averaging temp.')
    return parser

if __name__ == '__main__':
    main()
