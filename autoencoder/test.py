import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from autoencoder.model import AutoEncoder


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
    else:
        sys.exit("Testing interrupted! No model.")
    return model


def test(model_path, evaluation_path, results_path):
    """
    Use the trained model to restore the image.

    Args:
        model_path: The trained model.
        evaluation_path: The path to the evaluation data.
        results_path: The path to save the restored images.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder = load_model(device=device,model_path=model_path)
    autoencoder.eval()
    criterion = nn.MSELoss()

    test_loss_avg, num_batches = 0, 0


    evaluation_data = ImageFolder(root=evaluation_path, transform=ToTensor())
    original_data = Subset(evaluation_data, range(0, int(len(evaluation_data)/2)))
    corrupted_data = Subset(evaluation_data, range(int(len(evaluation_data)/2), len(evaluation_data)))

    original_dataloader = DataLoader(original_data, batch_size=32)
    corrupted_dataloader = DataLoader(corrupted_data, batch_size=32)
    
    batches = len(original_dataloader)
    print(batches)
    for i in range(batches):
        # get the next batch
        images, images = next(iter(original_dataloader)), next(iter(corrupted_dataloader))
    
        with torch.no_grad():

            # send the images to the device
            images : torch.Tensor = images[i].to(device)
            corrupted_images : torch.Tensor = corrupted_images[i].to(device)

            # image_batch = image_batch.to(device)

            # autoencoder reconstruction
            # image_batch_recon = autoencoder.forward(image_batch)

            # forward pass
            output = autoencoder.forward(corrupted_images)

            rand_tensor= torch.rand(64, 3,28,28) 

            # img1 = rand_tensor[0]
            # # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
            # save_image(img1, str(i) + '.png')


            for j in range(output.shape[0]):
                        fig = plt.figure(figsize=(50,25))
                        
                        ax = fig.add_subplot(1, 3, 1)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        imgplot = plt.imshow(np.transpose(images[j].cpu().detach().numpy(), (1, 2, 0)))
                        ax.set_title('Original', fontsize=20)

                        ax = fig.add_subplot(1, 3, 2)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        imgplot = plt.imshow(np.transpose(corrupted_images[j].cpu().detach().numpy(), (1, 2, 0)))
                        ax.set_title('Before', fontsize=20)
                        # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
                        ax = fig.add_subplot(1, 3, 3)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        imgplot = plt.imshow(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)))
                        # imgplot.set_clim(0.0, 0.7)
                        ax.set_title('After', fontsize=20)
                        # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')


                        # save_image(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)), str(i) + '.png')
                        # plt.imshow(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)))
                        # plt.show()

            # reconstruction error
            loss = criterion(output, images)

            test_loss_avg += loss.item()
            num_batches += 1
    
    test_loss_avg /= num_batches
    print('Average reconstruction error: %f' % (test_loss_avg))
    pass
