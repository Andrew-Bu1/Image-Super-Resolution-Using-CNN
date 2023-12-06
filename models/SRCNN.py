import torch
import torch.nn as nn

from utils.data import sort_image, proccess_image
from modules import constants as const


class SRCNN(nn.Module):
    def __init__(self, mode=None, model_dir=None):
        super().__init__()

        self.device = const.DEFAULT_DEVICE
        self.to(self.device)

        # The first convolutional layer with 9x9 kernel and 64 feature maps
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, padding=4)
        # The second convolutional layer with 1x1 kernel and 32 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1)
        # The third convolutional layer with 5x5 kernel and 3 feature maps
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=5, padding=2)
        # The ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the first convolutional layer and ReLU
        x = self.relu(self.conv1(x))
        # Apply the second convolutional layer and ReLU
        x = self.relu(self.conv2(x))
        # Apply the third convolutional layer
        x = self.conv3(x)
        # Return the output
        return x

    def run_train(self, **kwargs):
        proccess_image()
        sort_image()

        model = SRCNN()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(const.DEFAULT_EPOCHS):  # 50 epochs
            # for batch in train_loader:
            #     low_res_images, high_res_images = batch

            #     # Forward pass
            #     outputs = model(low_res_images)
            #     loss = criterion(outputs, high_res_images)

            #     # Backward pass and optimization
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            print("Model in training, with args: {}".format(kwargs))
