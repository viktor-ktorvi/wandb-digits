import torch
import wandb
import random
import numpy as np

from model_cnn import CNNModel
from utils import save_model, get_data
from train import train

if __name__ == '__main__':
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = get_data(slice=1000, train=True)
    val_dataset = get_data(slice=200, train=False)

    with wandb.init(project='test-cnn', mode='online'):
        model = CNNModel()
        model.to(device)
        print(model)

        train(model, train_dataset, val_dataset, device)

        model_input = torch.randn(
            (1, 1, wandb.config.input_dimensions['height'], wandb.config.input_dimensions['width']), device=device)
        save_model(model, model_input)

        y = model(model_input)
        print('Output shape: ', y.shape)
