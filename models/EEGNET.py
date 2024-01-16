import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, af=nn.ELU(alpha=1)):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, 1e-05)
        )
        self.DepthWiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, 1e-05),
            af,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, 1e-05),
            af,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, (7, 8)),
            nn.AvgPool2d(kernel_size=(15, 7), stride=(15, 7))
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.DepthWiseConv(x)
        x = self.separableConv(x)
        x = self.conv(x)
        x = x.squeeze()
        return x
