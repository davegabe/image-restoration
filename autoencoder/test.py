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
from tqdm import tqdm

from autoencoder.model import AutoEncoder, AutoEncoderDataset, load_model


def test(model_path, evaluation_path, results_path, batch_size: int = 32):
    """
    Use the trained model to restore the image.

    Args:
        model_path: The trained model.
        evaluation_path: The path to the evaluation data.
        results_path: The path to save the restored images.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder, epoch = load_model(device=device,model_path=model_path)
    if epoch == 1:
        sys.exit("Model missing")
    autoencoder.eval()
    criterion = nn.MSELoss()

    test_loss_avg = 0

    # load the training data images
    evaluation_data = AutoEncoderDataset(evaluation_path, ToTensor())
    evaluation_data_loader = DataLoader(evaluation_data, batch_size=batch_size)
    
    for i, (original_images, corrupted_images) in enumerate(tqdm(evaluation_data_loader)):
        with torch.no_grad():
            # send the images to the device
            original_images = original_images.to(device)
            corrupted_images = corrupted_images.to(device)

            # forward pass
            output = autoencoder.forward(corrupted_images)

            # print output images
            for j in range(output.shape[0]):
                fig = plt.figure(figsize=(50,25))
                
                ax = fig.add_subplot(1, 3, 1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                imgplot = plt.imshow(np.transpose(original_images[j].cpu().detach().numpy(), (1, 2, 0)))
                ax.set_title('Original', fontsize=20)

                ax = fig.add_subplot(1, 3, 2)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                imgplot = plt.imshow(np.transpose(corrupted_images[j].cpu().detach().numpy(), (1, 2, 0)))
                ax.set_title('Before', fontsize=20)

                ax = fig.add_subplot(1, 3, 3)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                imgplot = plt.imshow(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)))
                ax.set_title('After', fontsize=20)


                # save_image(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)), str(i) + '.png')
                # plt.imshow(np.transpose(output[j].cpu().detach().numpy(), (1, 2, 0)))
                # plt.show()

            # reconstruction error
            loss = criterion(output, original_images)

            test_loss_avg += loss.item()

            del original_images
            del corrupted_images
            del output
            del loss
            torch.cuda.empty_cache() # empty the cache
            
    
    test_loss_avg /= len(evaluation_data)
    print('Average reconstruction error: %f' % (test_loss_avg))
