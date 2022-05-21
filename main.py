import torch
import wandb

from model_cnn import CNNModel
from utils import save_model, get_data

if __name__ == '__main__':
    with wandb.init(project='test-cnn', mode='disabled'):
        model = CNNModel()
        print(model)

        dataset = get_data(slice=100)

        model_input = torch.randn(
            (1, 3, wandb.config.input_dimensions['height'], wandb.config.input_dimensions['width']))
        save_model(model, model_input)

        y = model(model_input)
        print('Output shape: ', y.shape)
