import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.nn import functional as F

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *

class Resnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Resnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.mid1 = DoubleConv(64, 128)
        self.mid2 = DoubleConv(128, 256)
        self.mid3 = DoubleConv(256, 128)
        self.mid4 = DoubleConv(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x) 
        x2 = self.mid1(x1)
        x3 = self.mid2(x2) 
        x4 = self.mid3(x3)
        x5 = self.mid4(x4)
        logits = self.outc(x5)
        return logits

class SmallUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SmallUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256// factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MediumUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MediumUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ExtraDeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ExtraDeepUNet, self).__init__()
        self.type(torch.complex64)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits

class Unet(nn.Module):
    """
    完整U-Net实现，包含科学初始化方案
    Args:
        in_chans (int): 输入通道数
        out_chans (int): 输出通道数
        chans (int): 第一层通道数，默认64
        num_pool_layers (int): 下采样次数，默认4
        drop_prob (float): Dropout概率，默认0.0
    """
    def __init__(self, in_chans, out_chans, chans=64, num_pool_layers=4, drop_prob=0.0):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        # 下采样路径
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        
        # 瓶颈层
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        # 上采样路径
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        # 最后一层
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image):
        stack = []
        output = image

        # 下采样
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        # 瓶颈层
        output = self.conv(output)

        # 上采样
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # 处理奇数尺寸
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # 右填充
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # 下填充
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        
        return output


class ConvBlock(nn.Module):
    """ 两次卷积+InstanceNorm+LeakyReLU """
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image):
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """ 转置卷积+InstanceNorm+LeakyReLU """
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image):
        return self.layers(image)

