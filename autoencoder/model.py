import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class AutoEncoder(nn.Module):
    def __init__(self, channels=3):
        super().__init__()

        # Number of nodes for internal layers
        nodes = np.linspace(1, 10, num=10)

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, channels, stride=2, padding=1, output_padding=1),
            nn.Conv2d(
                in_channels=32,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoderDataset(torch.utils.data.Dataset):
    """
    A custom dataset for the autoencoder.
    """

    def __init__(self, path, transform=ToTensor()):
        """
        Args:
            path: The path to the dataset.
            transform: The transform to apply to the images.
        """
        self.transform = transform
        self.original_images = []
        self.corrupted_images = []

        # Get the list of original images
        original_path = os.path.join(path, 'Original/')
        corrupted_path = os.path.join(path, 'Corrupted/')
        original_files = os.listdir(original_path)
        for file in original_files:
            self.original_images.append(os.path.join(original_path, file))
            extension = os.path.splitext(file)[1]
            file_name = os.path.basename(file[:-len(extension)])
            self.corrupted_images.append(os.path.join(
                corrupted_path, file_name + "_corrupted" + extension))

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, index):
        # load the original image
        original_image = Image.open(self.original_images[index]).convert('RGB')
        # load the corrupted image
        corrupted_image = Image.open(
            self.corrupted_images[index]).convert('RGB')
        return self.transform(original_image), self.transform(corrupted_image)


def load_model(device: torch.device, model_path: str):
    """
    Load the model if it exists, otherwise create a new one.

    Args:
        device: The device to run the model on.
        model_path: The path to the model.
    """
    model = AutoEncoder().to(device)
    epoch = 1
    if os.path.exists(model_path) and os.listdir(model_path):
        models = os.listdir(model_path)  # Get the list of models
        models.sort(key=lambda x: int(
            x.split('_')[1].split('.')[0]))  # Sort by epoch
        last_model = os.path.join(model_path, models[-1])  # Get the last model
        model.load_state_dict(torch.load(last_model))  # Load the last model
        epoch = int(last_model.split('_')[1].split(
            '.')[0]) + 1  # Get the epoch
    return model, epoch


def save_model(model, epoch: int, model_path: str):
    """
    Save the model.

    Args:
        model: The model to save.
        epoch: The epoch of the model.
        model_path: The path to the model.
    """
    os.makedirs(
        model_path, exist_ok=True)  # Create the model directory if it doesn't exist
    # The path to the model
    model_save = os.path.join(model_path, f"model_{epoch}.pth")
    torch.save(model.state_dict(), model_save)  # Save the model
    return model_save
