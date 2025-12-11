from torch import nn
import torch

class CNNNetwork(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super().__init__()

        self.conv1 = self._create_conv_block(1, 16)
        self.conv2 = self._create_conv_block(16, 32)
        self.conv3 = self._create_conv_block(32, 64)
        self.conv4 = self._create_conv_block(64, 128)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5) 

        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            linear_input_dim = x.view(1, -1).shape[1]

        self.linear = nn.Linear(linear_input_dim, num_classes)

    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        logits = self.linear(x)
        return logits