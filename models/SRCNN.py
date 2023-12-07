import torch
import torch.nn as nn

from modules import constants as const
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.metric import calculate_psnr
from utils.data import SRDataset


class SRCNN(nn.Module):
    def __init__(self, mode=None, model_dir=None):
        super().__init__()

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
        # Define image paths and data directories
        lr_image_dir = const.DEFAULT_LOW_RESOURCE_PATH
        hr_image_dir = const.DEFAULT_HIGH_RESOURCE_PATH if kwargs.get(
            "hr_image_dir") is None else kwargs["hr_image_dir"]

        # Create training and validation datasets
        train_data = SRDataset(lr_image_dir, hr_image_dir)
        val_data = SRDataset(lr_image_dir, hr_image_dir)

        # Create data loaders
        train_loader = DataLoader(
            train_data, batch_size=const.DEFAULT_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            val_data, batch_size=const.DEFAULT_BATCH_SIZE, shuffle=False)

        # Move model to device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.to(device)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(), lr=const.DEFAULT_LEARNING_RATE)
        criterion = nn.MSELoss()

        # Initialize training and validation loss lists
        train_loss_list = []
        val_loss_list = []

        # Create a writer for TensorBoard
        writer = SummaryWriter()

        for epoch in range(const.DEFAULT_EPOCHS):
            print(f'Epoch {epoch+1}/{const.DEFAULT_EPOCHS}:', end=' ')
            train_loss = 0

            # Training loop
            for i, (low_image_resource, high_image_resource) in enumerate(train_loader):
                # Move data to device
                low_image_resource = low_image_resource.to(device)
                high_image_resource = high_image_resource.to(device)

                # Forward pass and calculate loss
                outputs = model(low_image_resource)
                loss = criterion(outputs, high_image_resource)

                # Update model weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate training loss
                train_loss += loss.item()

            # Switch to evaluation mode
            model.eval()

            # Calculate validation loss
            val_loss = 0
            with torch.no_grad():
                for i, (low_image_resource, high_image_resource) in enumerate(val_loader):
                    # Move data to device
                    low_image_resource = low_image_resource.to(device)
                    high_image_resource = high_image_resource.to(device)

                    # Forward pass and calculate loss
                    outputs = model(low_image_resource)
                    loss = criterion(outputs, high_image_resource)

                    # Accumulate validation loss
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Track and print epoch-wise losses and write to TensorBoard
            train_loss_list.append(train_loss/len(train_loader))
            val_loss_list.append(val_loss)
            writer.add_scalar('Loss/Train', train_loss_list[-1], epoch)
            writer.add_scalar('Loss/Val', val_loss_list[-1], epoch)
            print(
                f"Train Loss: {train_loss_list[-1]}, Val Loss: {val_loss_list[-1]}")

        plt.plot(range(1, len(train_loss_list) + 1),
                 train_loss_list, label="Train Loss")
        plt.plot(range(1, len(val_loss_list) + 1),
                 val_loss_list, label="Validation Loss")
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.savefig('loss_plot.png')
        plt.close()

    def run_eval(self, data_loader, criterion):

        # Set model to evaluation mode
        self.eval()
        model = self.to(device=torch.device("cuda"))

        total_loss = 0
        total_psnr = 0
        for i, (low_image_resource, high_image_resource) in enumerate(data_loader):
            # Move data to device
            low_image_resource = low_image_resource.to(
                device=torch.device("cuda"))
            high_image_resource = high_image_resource.to(
                device=torch.device("cuda"))

            # Forward pass and calculate loss
            outputs = model(low_image_resource)
            loss = criterion(outputs, high_image_resource)
            psnr = calculate_psnr(outputs, high_image_resource)

            # Accumulate total loss
            total_loss += loss.item()
            total_psnr += psnr.item()
        # Calculate and return average loss
        avg_loss = total_loss / len(data_loader)
        avg_psnr = total_psnr / len(data_loader)
        return avg_loss, avg_psnr
