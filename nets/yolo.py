from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53
from nets.CBAM import cbam_block
from nets.spp import SpatialPyramidPooling
from .darknet import Mish


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
        # ("Mish", Mish()),
    ]))
#修改进行下采样惊醒融合
def conv2d1(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=2, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
        # ("Mish", Mish()),
    ]))
# ------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
# ------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m


# 获得三个卷积层
def make_three_conve(filter_list, in_filter):
    c = nn.Sequential(
        conv2d(in_filter, filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1),
    )
    return c


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained=False, cbma=False, spp=False):
        super(YoloBody, self).__init__()
        # ---------------------------------------------------#
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        # ---------------------------------------------------#
        self.backbone = darknet53(Mish1=False)
        # ----------------------------------
        # 添加空间特征金字塔
        # ----------------------------------
        self.spp1 = spp
        if spp:
            self.conv_1 = make_three_conve([512, 1024], 1024)
            self.spp = SpatialPyramidPooling()
            self.conv_2 = make_three_conve([1024, 512], 2048)

        # ----------------------------------
        # 添加空间-通道注意力机制
        # ----------------------------------
        self.cbma = cbma
        if cbma:
            self.up_sampling0 = cbam_block(1024)
            self.up_sampling1 = cbam_block(768)
            self.up_sampling2 = cbam_block(384)

        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        # ---------------------------------------------------#
        #   layers_out_filters : [64, 128, 256, 512, 1024]
        # ---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        # ------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        # ------------------------------------------------------------------------#
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        self.down_sample_conv=conv2d1(256,512,3)
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        self.conv=nn.Conv2d(1024,512,1)
    def forward(self, x):
        # ---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        # ---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)
        # ----------------------------------
        # 空间特征金字塔
        # ----------------------------------

        # ---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        # ---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        if self.spp1:
            x0 = self.conv_1(x0)
            P5 = self.spp(x0)
            x0 = self.conv_2(P5)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        if self.cbma:
            x0 = self.up_sampling0(x0)
        out0_branch = self.last_layer0[:5](x0)


        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        # ----------------------------------
        # 添加空间-通道注意力机制
        # ----------------------------------


        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        if self.cbma:
            x1_in = self.up_sampling1(x1_in)
        # ---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        # ---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)

        out0_branch1=self.down_sample_conv(out1_branch)
        out0_branch=torch.cat([out0_branch1,out0_branch],1)
        out0_branch=self.conv(out0_branch)
        out0 = self.last_layer0[5:](out0_branch)

        out1 = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        if self.cbma:
            x2_in = self.up_sampling2(x2_in)
        # ---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        # ---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2
