from autoencoder.model import (AutoEncoder, AutoEncoderDataset, load_model,
                               save_model)
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

isUsingTPU = False
if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm
    isUsingTPU = True


def train(training_path: str, model_path: str, epochs_save: int = 10, batch_size: int = 32, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train the autoencoder.

    Args:
        training_path: The path to the training data.
        model_path: The path to save the trained model.
    """
    # load the model if it exist, otherwise create a new one
    model, optimizer, criterion, epoch = load_model(
        device, model_path, forTraining=True)

    # load the training data images
    train_data = AutoEncoderDataset(training_path, ToTensor())
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    # summary(model, (3, 128, 128))

    # train the model
    train_model(model, train_data_loader, criterion, optimizer, epoch, epochs_save, model_path, device)


def train_model(model: AutoEncoder, train_data_loader: DataLoader, criterion: nn.L1Loss, optimizer: torch.optim.AdamW, epoch: int, epochs_save: int, model_path: str, device: torch.device):
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

            for i, (original_images, corrupted_images) in enumerate(tqdm(train_data_loader, desc=description)):
                # send the images to the device
                original_images: torch.Tensor = original_images.to(device)
                corrupted_images: torch.Tensor = corrupted_images.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                output_images = model.forward(corrupted_images)
                loss = criterion(output_images, original_images)
                lastLoss = loss.detach().cpu()
                loss.backward()
                if isUsingTPU:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()

                del original_images
                del corrupted_images
                del output_images
                del loss
                torch.cuda.empty_cache()

            # save the model every epochs_save times
            if(epoch % epochs_save == 0):
                saved_model = save_model(
                    model, criterion, optimizer, epoch, model_path)
                print(
                    f"[SAVED MODEL] Epoch: {epoch} | Saved to {saved_model}!")

                # if it's using TPU, save a copy of the model for inference on CPU/GPU
                if isUsingTPU:
                    xm.save(model_path + "lastTPUModel.xla")
            epoch += 1

    except KeyboardInterrupt:
        del original_images
        del corrupted_images
        del output_images
        del loss
        torch.cuda.empty_cache()  # clear the cache
        print("[STOPPED] Training interrupted!")
        return

    except Exception as e:
        # del original_images
        # del corrupted_images
        # del output_images
        # del loss
        torch.cuda.empty_cache()  # clear the cache
        print(f"[ERROR] {e}")
        return


def train_n_test(training_path: str, evaluation_path: str, model_path: str, epochs_save: int = 10, batch_size: int = 32, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train the model and test it.

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
    # load the model if it exist, otherwise create a new one
    model, optimizer, criterion, epoch = load_model(
        device, model_path, forTraining=True)

    # load the training data images
    train_data = AutoEncoderDataset(training_path, ToTensor())
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    # load the training data images
    evaluation_data = AutoEncoderDataset(evaluation_path, ToTensor())
    evaluation_data_loader = DataLoader(evaluation_data, batch_size=batch_size)

    lossPerEpochTrain = [] # list of avg loss per epoch for training
    lossPerEpochTrainLast = [] # last loss of the last epoch for training
    lossPerEpochTest = [] # list of avg loss per epoch for testing
    try:
        while True:
            if len(lossPerEpochTrain) > 0:
                description = f"[TRAIN] Epoch {epoch} | Loss traning last {lossPerEpochTrainLast[-1]:.5f} | Loss traning {lossPerEpochTrain[-1]:.5f} | Loss test {lossPerEpochTest[-1]:.5f}"
            else:
                description = f"[TRAIN] Epoch {epoch}"

            loss_avg = 0
            lastLoss = 1
            
            # Train cycle
            for i, (original_images, corrupted_images) in enumerate(tqdm(train_data_loader, desc=description)):
                # send the images to the device
                original_images: torch.Tensor = original_images.to(device)
                corrupted_images: torch.Tensor = corrupted_images.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                output_images = model.forward(corrupted_images)
                loss = criterion(output_images, original_images)
                loss_avg += loss.detach().cpu()
                lastLoss = loss.detach().cpu()
                loss.backward()
                optimizer.step()

                del original_images
                del corrupted_images
                del output_images
                del loss
                torch.cuda.empty_cache()

            # append last loss to array
            lossPerEpochTrainLast.append(lastLoss)
            # append avg loss to array
            lossPerEpochTrain.append(loss_avg / len(train_data_loader))


            loss_avg = 0

            # Test cycle
            for i, (original_images, corrupted_images) in enumerate(tqdm(evaluation_data_loader, desc=f"[TEST] Using model from epoch {epoch}")):
                with torch.no_grad():
                    # send the images to the device
                    original_images = original_images.to(device)
                    corrupted_images = corrupted_images.to(device)

                    # forward pass, permute and reshape
                    output_images = model.forward(corrupted_images)

                    # reconstruction error
                    loss = criterion(output_images, original_images)

                    loss_avg += loss.detach().cpu()

                    del original_images
                    del corrupted_images
                    del output_images
                    del loss
                    torch.cuda.empty_cache()  # empty the cache

            # append avg loss to array
            lossPerEpochTest.append(loss_avg / len(evaluation_data_loader))

            # save the model every epochs_save times
            if(epoch % epochs_save == 0):
                saved_model = save_model(
                    model, criterion, optimizer, epoch, model_path)
                print(
                    f"[SAVED MODEL] Epoch: {epoch} | Saved to {saved_model}!")
                
                # plot lossPerEpochTrain and lossPerEpochTest as a function of epoch
                plt.plot(lossPerEpochTrain, label="Train loss")
                plt.plot(lossPerEpochTest, label="Train epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.ylim(0, 1)
                plt.xlim(0, epoch)
                plt.legend()
                plt.show()

                # plot lossPerEpochTrainLast and lossPerEpochTest as a function of epoch
                plt.plot(lossPerEpochTrainLast, label="Train loss")
                plt.plot(lossPerEpochTest, label="Train epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.ylim(0, 1)
                plt.xlim(0, epoch)
                plt.legend()
                plt.show()
            epoch += 1

    except KeyboardInterrupt:
        del original_images
        del corrupted_images
        del output_images
        del loss
        torch.cuda.empty_cache()  # clear the cache
        print("[STOPPED] Training interrupted!")
        return

    except Exception as e:
        # del original_images
        # del corrupted_images
        # del output_images
        # del loss
        torch.cuda.empty_cache()  # clear the cache
        print(f"[ERROR] {e}")
        return