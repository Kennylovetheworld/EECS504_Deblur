import torch
from torch import nn
from torchvision.ops import DeformConv2d
from util import *

class ResConvBlock(nn.Module):
    def __init__(self, channels, stride, dilation):
        super(ResConvBlock, self).__init__()
        self.kernel_size = 3    # Fixed kernel size throughout the feature extractor
        # We are upsampling the input with padding = dialtion to make the residual/input and
        #   the output match in size. Though less likely, one other possibility is to downsample
        #   the residual, which fails to preserve the resolution.
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, self.kernel_size, stride, dilation, dilation),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, self.kernel_size, stride, dilation, dilation),
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
        self.conv2 = ResConvBlock(hidden_channels[0], 1, 1)
        self.conv3 = ResConvBlock(hidden_channels[0], 1, 1)
        self.conv4 = nn.Sequential( # downsampling by half here
            nn.Conv2d(hidden_channels[0], hidden_channels[1], self.kernel_size, 2, 1, 1),
            nn.ReLU()
        )
        self.conv5 = ResConvBlock(hidden_channels[1], 1, 1)
        self.conv6 = ResConvBlock(hidden_channels[1], 1, 1)
        self.conv7 = ResConvBlock(hidden_channels[1], 1, 2)
        self.conv8 = ResConvBlock(hidden_channels[1], 1, 2)
        self.conv9 = ResConvBlock(hidden_channels[1], 1, 4)
        self.conv10= ResConvBlock(hidden_channels[1], 1, 4)
        self.conv11= nn.Sequential(
            nn.Conv2d(hidden_channels[1]*4, hidden_channels[2], self.kernel_size, 1, 1, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        out1 = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        out2 = self.conv6(self.conv5(out1))
        out3 = self.conv8(self.conv7(out2))
        out4 = self.conv10(self.conv9(out3))
        out = self.conv11(torch.cat((out1, out2, out3, out4), dim=1))
        return out

class DeformConvBlock(nn.Module):
    def __init__(self, channels_conv=64, channels_deform=64, num_conv=1, use_act=False):
        super(DeformConvBlock, self).__init__()
        # The conv and defromConv should have the same stride, kernel size and dilation so
        #   that the output of conv matches the size of output of deformConv and therefore can
        #   serve as the offset
        self.kernel_size = 3
        self.conv = nn.Sequential()
        for i in range(num_conv - 1):
            if use_act:
                self.conv.add_module("act %d"%i, nn.PReLU(channels_conv))
            self.conv.add_module("conv %d"%i, nn.Conv2d(channels_conv, channels_conv, self.kernel_size, 1, 1, 1))
        if use_act:
            self.conv.add_module("act_final", nn.PReLU(channels_conv))
        self.conv.add_module("conv_final", nn.Conv2d(channels_conv, 3*3*3, self.kernel_size, 1, 1, 1))
        self.deformConv = DeformConv2d(channels_deform, channels_deform, self.kernel_size, 1, 1, 1)
    
    def forward(self, content_feats, blur_feats):
        ow = self.conv(blur_feats)
        offset = ow[:,:2*3*3]
        weight = ow[:.2*3*3:] # Reimplement deformConv?
        out = self.deformConv(content_feats, offset)
        return out, offset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = FeatureExtractor()
        self.neck1 = DeformConvBlock()
        self.neck2 = DeformConvBlock(channels_conv=18, num_conv=2, use_act=True)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1, dilation=1),   # Upsampling
            nn.PReLU(64),
            nn.Conv2d(64, 3, 3, 1, 1, 1)
        )
        self.shortcut = nn.Identity()
        self.apply(weights_init)
    
    def forward(self, x):
        residual = self.shortcut(x)
        feats = self.backbone(x)
        rst1, offset = self.neck1(feats[:,:64],feats[:,64:])
        rst2, _ = self.neck2(rst1, offset)
        out = residual + self.head(rst2)
        return out, offset
