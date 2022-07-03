import os

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


class AutoEncoder(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()

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
            # nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(
            #     in_channels=256,
            #     out_channels=512,
            #     kernel_size=(3, 3),
            #     stride=(1, 1),
            #     padding=(1, 1)
            # ),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=512,
            #     out_channels=256,
            #     kernel_size=(3, 3),
            #     stride=(1, 1),
            #     padding=(1, 1)
            # ),
            # nn.ReLU(),
            # nn.ConvTranspose2d(256, 256, 3, stride=2,
            #                    padding=1, output_padding=1),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            # nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
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
        self.corrupted_images = []
        self.original_path = os.path.join(path, 'Original/')

        # Get the list of original images
        corrupted_path = os.path.join(path, 'Corrupted/')
        corrupted_files = os.listdir(corrupted_path)
        self.corrupted_images = [os.path.join(corrupted_path, file) for file in corrupted_files]

    def __len__(self):
        return len(self.corrupted_images)

    def __getitem__(self, index):
        # load the corrupted image
        corrupted_image = Image.open(self.corrupted_images[index]).convert('RGB')
        # get file extension
        file_extension = os.path.splitext(self.corrupted_images[index])[1]
        # get basename of the file
        file_name = os.path.basename(self.corrupted_images[index])
        # get the original image file
        original_image_file = self.original_path + file_name.split('_corrupted')[0] + file_extension
        # load the original image
        original_image = Image.open(original_image_file).convert('RGB')
        return self.transform(original_image), self.transform(corrupted_image)


def load_model(device: torch.device, model_path: str, forTraining: bool = True):
    """
    Load the model if it exists, otherwise create a new one.

    Args:
        device: The device to run the model on.
        model_path: The path to the model.
    """
    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    epoch = 1
    if os.path.exists(model_path) and os.listdir(model_path):
        models = os.listdir(model_path)  # Get the list of models
        models = [model for model in models if model.endswith('.pth')]  # Filter only .pth files
        models.sort(key=lambda x: int(
            x.split('_')[1].split('.')[0]))  # Sort by epoch
        last_model = os.path.join(model_path, models[-1])  # Get the last model

        # Load the last model
        checkpoint = torch.load(last_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        criterion = checkpoint['loss']
        if (forTraining):
            model.train()
        else:
            model.eval()
    return model, optimizer, criterion, epoch


def save_model(model: AutoEncoder, criterion: nn.L1Loss, optimizer: torch.optim.AdamW, epoch: int, model_path: str):
    """
    Save the model for inference and training.

    Args:
        model: The model to save.
        epoch: The epoch of the model.
        model_path: The path to the model.
    """
    os.makedirs(model_path, exist_ok=True)  # Create the model directory if it doesn't exist
    # The path to the model
    model_save = os.path.join(model_path, f"model_{epoch}.pth")
    # Save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, model_save)
    return model_save
