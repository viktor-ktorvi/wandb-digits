import torch
import wandb

from model_cnn import CNNModel
from utils import save_model

if __name__ == '__main__':
    with wandb.init(project='test-cnn', mode='online'):
        model = CNNModel()
        print(model)
        model_input = torch.randn((100, 100, 3))
        save_model(model, model_input)
