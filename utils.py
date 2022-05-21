import yaml
from argparse import Namespace
import torch
import datetime
import os
import wandb


def load_default_config_if_needed(config_path, wandb_config_object):
    if callable(wandb_config_object):
        with open(config_path, 'r') as stream:
            config_file = yaml.safe_load(stream)
        config = {}
        for key in config_file:
            val = config_file[key]['value']
            # try to convert lists of values into tensors here
            if isinstance(val, list):
                try:
                    val = torch.tensor(val)
                except:
                    pass
            config[key] = val
        return Namespace(**config)
    else:
        return Namespace(**wandb_config_object._items)


def save_model(model, model_input):
    model_folder = wandb.config.model_folder
    os.makedirs(model_folder, exist_ok=True)
    datetime_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # timestamp
    path = os.path.join(model_folder, 'state_dict_and_config_' + datetime_now)

    model.save_state_and_config(path + '.p')
    torch.onnx.export(model, model_input, path + '.onnx')

    wandb.save(path + '.p')
    wandb.save(path + '.onnx')
