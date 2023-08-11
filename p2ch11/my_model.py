import math

from torch import nn as nn
from util.logconf import logging

log = loggin.getLogger(__name__)
log.setLevel(logging.DEBUG)

# LunaModel を作成


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BachNorm3d(1)  # 平均を0, 標準偏差を1にする (引数: チャネル数)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels*2)
        self.block3 = LunaBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaBlock(conv_channels*4, conv_channels*8)

        self.head_linear = nn.Linear(1152, 2)  # 全結合層
        self.head_softmax = nn.Softmax(dim=1)  # 出力を確率に変換する

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(block_out.size(0), -1)  # 1次元に変換
        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)

        self._init_weights()  # パラメータを初期化する

    # パラメータを初期化する
    def _init_weights(self):
        for m in self.modules():  # このインスタンスのモジュールを取得
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTrasnpose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=1, mode='fan_out', nonlinearity='relu')  # Heの初期化
                if m.bias is not None:  # バイアスがある場合
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        m.weight.data)
                    bound = 1/math.sqrt(fan_out)  # バイアスの初期値の上限
                    nn.init.normal_(m.bias, -bound, bound)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()  # nn.Module の __init__ を呼び出す

        self.conv1 = nn.Conv3d(in_channels, conv_channels,
                               kernle_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLu(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, conv_channels,
                               kernle_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLu(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
