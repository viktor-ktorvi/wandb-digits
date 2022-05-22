import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train(model, train_dataset, val_dataset, device):
    """
    Train loop.
    :param model: torch model
    :param train_dataset: train dataset
    :param val_dataset: validation dataset
    :param device: torch device
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=wandb.config.learning_rate,
                                 weight_decay=wandb.config.weight_decay)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
    #                                                        factor=config.scheduler_factor,
    #                                                        patience=config.scheduler_patience,
    #                                                        threshold=config.scheduler_threshold,
    #                                                        verbose=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=wandb.config.batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(wandb.config.epochs)):
        process_epoch(epoch, model, optimizer, criterion, train_dataloader, dataset_type='train', device=device)
        process_epoch(epoch, model, optimizer, criterion, val_dataloader, dataset_type='validation', device=device)

        wandb.log({
            'epoch': epoch,
            'learning rate': wandb.config.learning_rate
        }, step=epoch)


def process_epoch(epoch, model, optimizer, criterion, dataloader, dataset_type, device):
    """
    One epoch for the specified dataloader.
    :param epoch: current epoch
    :param model:
    :param optimizer:
    :param criterion:
    :param dataloader:
    :param dataset_type: A string used in the logging to specify which dataset is being processed.
    :param device:
    :return:
    """
    epoch_loss = 0
    epoch_acc = 0
    for _, (images, labels) in enumerate(dataloader):
        loss, acc = process_batch(images.to(device), labels.to(device), model, optimizer, criterion)

        epoch_loss += loss / wandb.config.batch_size
        epoch_acc += acc

    wandb.log(
        {dataset_type + "/epoch": epoch,
         dataset_type + "/loss": epoch_loss / len(dataloader),
         dataset_type + "/accuracy": epoch_acc / len(dataloader)},
        step=epoch)


def process_batch(images, labels, model, optimizer, criterion):
    """
    Inference and update, as well as accuracy calculation.
    :param images:
    :param labels:
    :param model:
    :param optimizer:
    :param criterion:
    :return: loss and accuracy
    """
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    pred = torch.argmax(outputs, dim=1)
    acc = accuracy_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())

    return loss, acc
