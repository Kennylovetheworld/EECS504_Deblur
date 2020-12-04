import torch
from torch import nn

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation):
        super(ResConvBlock, self).__init__()
        self.kernel_size = 3    # Fixed kernel size throughout the feature extractor
        # We are upsampling the input with padding = dialtion to make the residual/input and
        #   the output match in size. Though less likely, one other possibility is to downsample
        #   the residual, which fails to preserve the resolution.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, stride, dilation, dilation),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, self.kernel_size, stride, dilation, dilation),
        )
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x) + residual
        return out

class FeatureExtractor(nn.Module):
    """
    The feature extractor
    """
    def __init__(self, in_channels=3, hidden_channels = [32, 64, 128], ):
        super().__init__()
        self.kernel_size = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], self.kernel_size, 1, 1, 1),
            nn.ReLU()
        )
        self.conv2 = ResConvBlock(hidden_channels[0], hidden_channels[0], 1, 1)
        self.conv3 = ResConvBlock(hidden_channels[0], hidden_channels[0], 1, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], self.kernel_size, 2, 1, 1),
            nn.ReLU()
        )   # downsampling here
        self.conv5 = ResConvBlock(hidden_channels[1], hidden_channels[1], 1, 1)
        self.conv6 = ResConvBlock(hidden_channels[1], hidden_channels[1], 1, 1)
        self.conv7 = ResConvBlock(hidden_channels[1], hidden_channels[1], 1, 2)
        self.conv8 = ResConvBlock(hidden_channels[1], hidden_channels[1], 1, 2)
        self.conv9 = ResConvBlock(hidden_channels[1], hidden_channels[1], 1, 4)
        self.conv10= ResConvBlock(hidden_channels[1], hidden_channels[1], 1, 4)
        self.conv11= nn.Sequential(
            nn.Conv2d(hidden_channels[1]*4, hidden_channels[2], self.kernel_size, 1, 1, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        out1 = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        out2 = self.conv6(self.conv5(out1))
        out3 = self.conv8(self.conv7(out2))
        out4 = self.conv10(self.conv9(out3))
        out = self.conv11(torch.cat((out1, out2, out3, out4)))
        return out 