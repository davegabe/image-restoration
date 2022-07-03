import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from autoencoder.model import AutoEncoderDataset, load_model

isUsingTPU = False
if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm
    isUsingTPU = True


def test(model_path: str, evaluation_path: str, results_path: str, batch_size: int = 32, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Use the trained model to restore the image.

    Args:
        model_path: The trained model.
        evaluation_path: The path to the evaluation data.
        results_path: The path to save the restored images.
    """
    autoencoder, optimizer, criterion, epoch = load_model(device, model_path, forTraining=False)
    if epoch == 1:
        sys.exit("Model missing")

    test_loss_avg = 0

    # load the training data images
    evaluation_data = AutoEncoderDataset(evaluation_path, ToTensor())
    evaluation_data_loader = DataLoader(evaluation_data, batch_size=batch_size)

    for i, (original_images, corrupted_images) in enumerate(tqdm(evaluation_data_loader, desc=f"Using model from epoch {epoch-1}")):
        with torch.no_grad():
            # send the images to the device
            original_images = original_images.to(device)
            corrupted_images = corrupted_images.to(device)

            # forward pass, permute and reshape
            output_images = autoencoder.forward(corrupted_images)

            # print output images
            for j in range(output_images.shape[0]):
                fig = plt.figure(figsize=(50, 25))

                ax = fig.add_subplot(1, 3, 1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                imgplot = plt.imshow(np.transpose(
                    original_images[j].cpu().detach().numpy(), (1, 2, 0)))
                ax.set_title('Original', fontsize=20)

                ax = fig.add_subplot(1, 3, 2)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                imgplot = plt.imshow(np.transpose(
                    corrupted_images[j].cpu().detach().numpy(), (1, 2, 0)))
                ax.set_title('Before', fontsize=20)

                ax = fig.add_subplot(1, 3, 3)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                imgplot = plt.imshow(np.transpose(
                    output_images[j].cpu().detach().numpy(), (1, 2, 0)))
                ax.set_title('After', fontsize=20)

                name = str(i) + "_" + str(j)+".jpg"

                plt.savefig(results_path + name)
                # plt.draw()
                plt.clf()
                plt.close("all")

            # reconstruction error
            loss = criterion(output_images, original_images)

            test_loss_avg += loss.item()

            del original_images
            del corrupted_images
            del output_images
            del loss
            torch.cuda.empty_cache()  # empty the cache

    test_loss_avg /= len(evaluation_data)
    print('Average reconstruction error: %f' % (test_loss_avg))
