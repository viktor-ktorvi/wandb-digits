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

        self.convs = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3 if i == 0 else self.config.channels,
                                out_channels=self.config.channels,
                                kernel_size=(self.config.kernel_size, self.config.kernel_size),
                                padding=self.config.kernel_size // 2),
                torch.nn.ReLU()
            ) for i in range(self.config.num_layers - 1)])

        self.final_conv = torch.nn.Conv2d(in_channels=self.config.channels,
                                          out_channels=1,
                                          kernel_size=(self.config.kernel_size, self.config.kernel_size),
                                          padding=self.config.kernel_size // 2)

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)

        x = self.final_conv(x)
        return x

    def save_state_and_config(self, savepath):
        states = self.state_dict()
        torch.save((states, self.config), savepath)
