from torch import nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AutoEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, layers=3):
        super().__init__()

        # Encoder (convolutional layers)
        sequential = nn.Sequential()
        for i in range(layers):
            sequential.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=input_size if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            sequential.add_module(
                f"relu_{i}",
                nn.ReLU()
            )
        self.encoder = sequential

        # Decoder (transposed convolutional layers)
        sequential = nn.Sequential()
        for i in range(layers):
            sequential.add_module(
                "tconv_{i}",
                nn.ConvTranspose2d(
                    in_channels=hidden_size,
                    out_channels=input_size if i == layers - 1 else hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            sequential.add_module(
                "relu_{i}",
                nn.ReLU()
            )
        self.decoder = sequential

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
