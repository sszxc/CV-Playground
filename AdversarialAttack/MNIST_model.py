# Author: Xuechao Zhang
# Date: May 18th, 2022
# Description:
#   Use Net() and Data() to get MNIST model and dataloader for FGSM attack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class Net(nn.Module):
    """
    Load a pre-trained LeNet Model
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Load the pretrained model
        # Download it from https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h
        self.load_state_dict(torch.load("lenet_mnist_model.pth", map_location='cpu'))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def Data():
    """
    Load the MNIST dataset
    """
    # 会自动下载MNIST数据集
    return torch.utils.data.DataLoader(
                datasets.MNIST('./', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),])),
                batch_size=1, shuffle=True)