import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from autoencoder.model import AutoEncoder


def train(training_path: str, model_path: str, epochs_save: int = 10, batch_size: int = 32):
    """
    Train the autoencoder.

    Args:
        training_path: The path to the training data.
        model_path: The path to save the trained model.
    """

    # select the device to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the model if it exist, otherwise create a new one
    model, epoch = load_model(device, model_path)

    # define the loss function
    criterion = nn.MSELoss()

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # load the training data images
    train_data = ImageFolder(root=training_path, transform=ToTensor())
    original_data = Subset(train_data, range(0, int(len(train_data)/2)))
    corrupted_data = Subset(train_data, range(int(len(train_data)/2), len(train_data)))

    original_dataloader = DataLoader(original_data, batch_size=batch_size)
    corrupted_dataloader = DataLoader(corrupted_data, batch_size=batch_size)
    
    # train the model
    train_model(model, original_dataloader, corrupted_dataloader, criterion, optimizer,
                epoch, epochs_save, model_path, device)


def load_model(device: torch.device, model_path: str):
    """
    Load the model if it exists, otherwise create a new one.

    Args:
        device: The device to run the model on.
        model_path: The path to the model.
    """
    model = AutoEncoder(3, 8, 1).to(device)
    epoch = 1
    if os.path.exists(model_path) and os.listdir(model_path):
        models = os.listdir(model_path)
        model.load_state_dict(torch.load(os.path.join(model_path, models[-1])))
        epoch = int(models[-1].split(".")[0].split("_")[-1]) + 1
    return model, epoch


def train_model(model: AutoEncoder, original_dataloader: DataLoader, corrupted_dataloader: DataLoader, criterion: nn.MSELoss, optimizer: torch.optim.AdamW, epoch: int, epochs_save: int, model_path: str, device: torch.device):
    """
    Train the model.

    Args:
        model: The model to train.
        original_dataloader: The dataloader to use for calculate loss.
        corrupted_dataloader: The dataloader to use for training.
        criterion: The loss function to use.
        optimizer: The optimizer to use.
        epoch: The current epoch.
        epochs_save: The number of epochs to save the model after.
        model_path: The path to save the model to.
        device: The device to run the model on.
    """
    try:
        while True:
            batches = len(original_dataloader)
            for i in range(batches):
                # get the next batch
                images, corrupted_images = next(iter(original_dataloader)), next(iter(corrupted_dataloader))

                # send the images to the device
                images : torch.Tensor = images[i].to(device)
                corrupted_images : torch.Tensor = corrupted_images[i].to(device)

                # forward pass
                output = model.forward(corrupted_images)

                # calculate the loss
                loss = criterion(output, images)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print the loss
                print(f"Epoch: {epoch} | Batch: {i+1}/{batches} | Loss: {loss.item():.4f}")

                # print output images
                if(epoch % epochs_save == 0):
                    for j in range(output.shape[0]):
                        plt.imshow(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)))
                        plt.show()

            # save the model every epochs_save times
            if(epoch % epochs_save == 0):
                os.makedirs(model_path, exist_ok=True)
                model_save = os.path.join(model_path, f"model_{epoch}.pth")
                torch.save(model.state_dict(), model_save)
                print(f"Epoch: {epoch} | Saved model to {model_save}!")
            epoch += 1

    except KeyboardInterrupt:
        print('Training interrupted!')
