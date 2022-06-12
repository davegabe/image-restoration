import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchsummary import summary
from torchvision import models
from torchvision.transforms import ToTensor
from tqdm import tqdm

from autoencoder.model import (AutoEncoder, AutoEncoderDataset, load_model,
                               save_model)


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

    # summary(model, (3, 512, 512))

    # define the loss function
    criterion = nn.MSELoss()

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # load the training data images
    train_data = AutoEncoderDataset(training_path, ToTensor())
    train_data_loader = DataLoader(train_data, batch_size=batch_size)

    # train the model
    train_model(model, train_data_loader, criterion, optimizer,
                epoch, epochs_save, model_path, device)

def train_model(model: AutoEncoder, train_data_loader: DataLoader, criterion: nn.MSELoss, optimizer: torch.optim.AdamW, epoch: int, epochs_save: int, model_path: str, device: torch.device):
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
        lastLoss = 1
        while True:
            if lastLoss != 1:
                description = f"Epoch {epoch} | Loss {lastLoss}"
            else:
                description = f"Epoch {epoch}"
                
            for i, (original_image, corrupted_image) in enumerate(tqdm(train_data_loader, desc=description)):
                # send the images to the device
                original_image: torch.Tensor = original_image.to(device)
                corrupted_image: torch.Tensor = corrupted_image.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
       
                # forward + backward + optimize
                original_image = torch.permute(original_image, (2, 3, 0, 1)).view(512 * 512, 4, 3)
                corrupted_image = torch.permute(corrupted_image, (2, 3, 0, 1)).view(512 * 512, 4, 3)

                output = model.forward(corrupted_image)
                loss = criterion(output, original_image)
                lastLoss = loss.detach().cpu()
                loss.backward()
                optimizer.step()

                del original_image
                del corrupted_image
                del output
                del loss
                torch.cuda.empty_cache()

            # # print output images
            # if(epoch % epochs_save == 0):
            #     for j in range(output.shape[0]):
            #         print(f"#### {j} Image ####")
            #         print(f"Min: {output[j].cpu().min()}")
            #         print(f"Max: {output[j].cpu().max()}")
            #         plt.imshow(np.transpose(
            #             output[j].cpu().detach().numpy(), (1, 2, 0)))
            #         plt.show()

            # save the model every epochs_save times
            if(epoch % epochs_save == 0):
                saved_model = save_model(model, epoch, model_path)
                print(f"[SAVED MODEL] Epoch: {epoch} | Saved to {saved_model}!")
            epoch += 1

    except KeyboardInterrupt:
        del original_image
        del corrupted_image
        del output
        del loss
        torch.cuda.empty_cache() # clear the cache
        print("[STOPPED] Training interrupted!")
    
    except Exception as e:
        # del original_image
        # del corrupted_image
        # del output
        # del loss
        torch.cuda.empty_cache() # clear the cache
        print(f"[ERROR] {e}")
