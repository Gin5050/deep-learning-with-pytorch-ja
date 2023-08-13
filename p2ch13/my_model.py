import math
import random
from collections import namedtuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):  # kwargsは辞書型で受け取る
        super().__init__()

        # バッチ正規化を追加, nn.BatchNorm2dにはチャネル数を指定する
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        # UNetのインスタンスを作成
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()  # sigmoid関数により出力を0から1の範囲に収める

        self._init_weights()  # 独自の重みの初期化

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }

        for m in self.modules():  # self.modules()はネットワークの全てのモジュールを返す
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu')  # Heの初期化
                if m.bais is not None:  # もしバイアスがあれば初期化
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def formard(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)  # バッチ正規化
        un_output = self.unet(bn_output)  # UNetを実行
        fn_output = self.final(un_output)  # sigmoid関数により出力を0から1の範囲に収める
        return fn_output
