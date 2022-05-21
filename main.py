import torch
import wandb

from model_cnn import CNNModel
from utils import save_model, get_data

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_data(slice=100)

    with wandb.init(project='test-cnn', mode='disabled'):
        model = CNNModel()
        model.to(device)
        print(model)

        model_input = torch.randn(
            (1, 3, wandb.config.input_dimensions['height'], wandb.config.input_dimensions['width']), device=device)
        save_model(model, model_input)

        y = model(model_input)
        print('Output shape: ', y.shape)
