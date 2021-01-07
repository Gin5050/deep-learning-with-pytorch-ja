import math
import random
import torch
from torch import nn as nn
import torch.nn.functional as F

import numpy as np

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class ClassificationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        print('====== 1 ======')
        print(input_g.shape)
        transform_t = self._build2dTransformMatrix()
        print('====== 2 ======')
        print(transform_t.shape)
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        print('====== 3 ======')
        print(transform_t.shape)
        transform_t = transform_t.to(input_g.device, torch.float32)
        print(transform_t[:3].unsqueeze(0).expand(
            input_g.size(0), -1, -1).shape)
        affine_t = F.affine_grid(transform_t[:3].unsqueeze(0).expand(input_g.size(0), -1, -1),
                                 input_g.size(), align_corners=False)
        # affine_t = F.affine_grid(transform_t[:, :2],
        #                          input_g.size(), align_corners=False)
        print('====== 4 ======')
        print(affine_t.shape)
        augmented_input_g = F.grid_sample(input_g,
                                          affine_t, padding_mode='border',
                                          align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                                          affine_t, padding_mode='border',
                                          align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(4, dtype=torch.float32)

        for i in range(3):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[3, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            # angle_rad = random.random() * math.pi * 2
            # s = math.sin(angle_rad)
            # c = math.cos(angle_rad)

            # rotation_t = torch.tensor([
            #     [c, -s, 0],
            #     [s, c, 0],
            #     [0, 0, 1]])
            angle_rad = random.random() * np.pi * 2
            s = np.sin(angle_rad)
            c = np.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32)

            transform_t @= rotation_t

        return transform_t


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
