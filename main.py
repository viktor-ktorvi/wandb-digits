import torch
import wandb

from model_cnn import CNNModel
from utils import save_model, get_data, set_random_seeds
from train import train

if __name__ == '__main__':
    with wandb.init(project='test-cnn', mode='online'):  # mode: ['online', 'disabled']
        # Ensure deterministic behavior
        set_random_seeds()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # datasets
        train_dataset = get_data(approx_size=wandb.config.train_dataset_size, train=True,
                                 dataset_name=wandb.config.dataset_name)
        val_dataset = get_data(approx_size=wandb.config.val_dataset_size, train=False,
                               dataset_name=wandb.config.dataset_name)

        # data
        input_channels, height, width = train_dataset[0][0].shape

        if hasattr(train_dataset.dataset, 'classes'):
            num_classes = len(train_dataset.dataset.classes)
        elif hasattr(train_dataset.dataset, 'targets'):
            num_classes = len(torch.tensor(train_dataset.dataset.targets).unique())
        else:
            raise ValueError("I don't know how to get the number of classes from this dataset!")

        wandb.config.update({'input_dimensions': {'height': height, 'width': width},
                             'input_channels': input_channels,
                             'num_classes': num_classes},
                            allow_val_change=True)

        # model
        if wandb.config.model_type == 'cnn':
            model = CNNModel()
        else:
            raise NotImplementedError("Model type {:s} isn't implemented".format(wandb.config.model_type))

        model.to(device)
        print(model)

        # train loop
        train(model, train_dataset, val_dataset, device)

        # saving the model
        model_input = val_dataset[0][0].unsqueeze(0).to(device)
        save_model(model, model_input)

        print('Input shape: ', model_input.shape)

        y = model(model_input)
        print('Output shape: ', y.shape)
