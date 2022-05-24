import yaml
from argparse import Namespace
import torch
import datetime
import os
import wandb
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np


def get_data(approx_size, train=True, dataset_name='mnist'):
    """
    Load the dataset. Code mostly taken form a wandb tutorial.
    :param approx_size: the approximate size of the subset of data. It's not going to be exactly that all the time because
    slicing is used, but it's close.
    :param train: True if you want the training set, False if validation.
    :param dataset_name: ['mnist', 'cifar10'] are implemented.
    :return: A subset of a dataset.
    """
    if dataset_name == 'mnist':
        # remove slow mirror from list of MNIST mirrors
        torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                              if not mirror.startswith("http://yann.lecun.com")]
        full_dataset = torchvision.datasets.MNIST(root="./data",
                                                  train=train,
                                                  transform=transforms.ToTensor(),
                                                  download=True)
    elif dataset_name == 'cifar10':
        full_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=train,
                                                    download=True,
                                                    transform=transforms.ToTensor())
    elif dataset_name == 'fashion-mnist':
        full_dataset = torchvision.datasets.FashionMNIST(root="./data",
                                                         train=train,
                                                         transform=transforms.ToTensor(),
                                                         download=True)
    elif dataset_name == 'cifar100':
        full_dataset = torchvision.datasets.CIFAR100(root="./data",
                                                     train=train,
                                                     transform=transforms.ToTensor(),
                                                     download=True)
    elif dataset_name == 'kmnist':
        full_dataset = torchvision.datasets.KMNIST(root="./data",
                                                   train=train,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

    else:
        raise NotImplementedError("Dataset {:s} isn't implemented".format(dataset_name))

    if approx_size < len(full_dataset):
        slice = len(full_dataset) // approx_size
    else:
        slice = 1

    #  equiv to slicing with [::slice]
    sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), slice))
    return sub_dataset


def set_random_seeds():
    """
    All the randomness I hope. Code from a wandb tutorial.
    :return:
    """
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


def load_default_config_if_needed(config_path, wandb_config_object):
    """
    Turning the config into an object. This code isn't mine, I don't know the details.

    :param config_path: where is config-defaults.yaml
    :param wandb_config_object: wandb.config
    :return: the config object I guess
    """
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
    """
    Save the model as a pickle file(I think) and as an ONNX file. Also adds a time stamp to the filename.
    :param model: torch model
    :param model_input: a data sample that goes into the model. It's needed for the ONNX part
    :return:
    """
    model_folder = wandb.config.model_folder
    os.makedirs(model_folder, exist_ok=True)
    datetime_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # timestamp
    path = os.path.join(model_folder, 'state_dict_and_config_' + datetime_now)

    model.save_state_and_config(path + '.p')
    torch.onnx.export(model, model_input, path + '.onnx')

    wandb.save(path + '.p')
    wandb.save(path + '.onnx')
