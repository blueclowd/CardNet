
import torch
import torch.nn as nn

class CardNet(torch.nn.Module):

    def __init__(self):

        super(CardNet, self).__init__()

        # input shape = 512 x 512

        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2))

        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2))

        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2))

        self.fc_1 = torch.nn.Linear(128*64*64, 1024)
        self.fc_2 = torch.nn.Linear(1024, 8)

    def forward(self, x):

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = x.contiguous().view(-1, 128 * 64 * 64)

        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


