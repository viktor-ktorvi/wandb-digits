import torch
import wandb
import datetime
import os

from utils import load_default_config_if_needed


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.config = load_default_config_if_needed(config_path='config-defaults.yaml',
                                                    wandb_config_object=wandb.config)

    def forward(self, x):
        return x

    def save_state_and_config(self, savepath):
        states = self.state_dict()
        torch.save((states, self.config), savepath)

