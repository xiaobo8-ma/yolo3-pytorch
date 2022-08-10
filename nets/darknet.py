import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------#
#   MISH激活函数
# -------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


"""
将残差结构里面的3x3卷据换成深度可分离卷积
"""


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        # 208x208x32->208x208x32
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),  # groups的意思就是是否进行分组卷积
        nn.BatchNorm2d(inp),  # inp是特征层的数量，对其进行归一化处理
        nn.ReLU6(inplace=False),  # inplace为true时对输入的张量进行改变（对原始变量进行覆盖），节省内存空间。
        # 为False时不做改变，对结果没有影响
        # 208x208x64
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False),
    )


# ---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上个一残差边
# ---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, Mish1=False):
        # 208x208x64  64  [32,64]
        super(BasicBlock, self).__init__()
        # 208x208x64->208x208x32
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        if Mish1:
            self.relu1 = Mish()
        else:
            self.relu1 = nn.LeakyReLU(0.1)
        # ------------------------------------------------
        # 将残差结构里面的3x3卷据换成深度可分离卷积
        # ------------------------------------------------
        self.dpc = conv_dw(planes[0], planes[1])
        # self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes[1])
        # # self.relu2 = nn.LeakyReLU(0.1)
        # if Mish1:
        #     self.relu2 = Mish()
        # else:
        #     self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dpc(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers, Mish1=False):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu1 = nn.LeakyReLU(0.1)
        if Mish1:
            self.relu1 = Mish()
        else:
            self.relu1 = nn.LeakyReLU(0.1)
        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0], Mish1=False)
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1], Mish1=False)
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2], Mish1=False)
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3], Mish1=False)
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4], Mish1=False)

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # a = m.kernel_size[0]
                # b = m.kernel_size[1]
                # c = m.out_channels
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # print(a, b, c, '=', n)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    # ---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks, Mish1=False):
        layers = []  # planes=[32,64] blocks=1  416x416x32->208x208x64
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        if Mish1:
            # self.relu1 = Mish()
            layers.append(("ds_relu", Mish()))
        else:

            layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 加入残差结构
        self.inplanes = planes[1]

        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return (out3, out4, out5)


def darknet53(Mish1=False):
    model = DarkNet([1, 2, 8, 8, 4], Mish1=False)
    return model


if __name__ == '__main__':
    print(darknet53())
