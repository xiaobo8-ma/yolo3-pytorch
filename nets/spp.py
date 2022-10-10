"""
空间金字塔池化模块
"""
import torch
from torch import nn


# ---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
# ---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat(features + [x], dim=1)

        return features


# class  SpatialPyramidPooling(nn.Module):
#     def __init__(self):
#         super(SpatialPyramidPooling, self).__init__()
#         self.pooling_size = [5, 9, 13]
#         self.pooling = nn.ModuleList([nn.MaxPool2d(pool, 1, pool // 2) for pool in self.pooling_size])
#
#     def forward(self, x):
#         features = [maxpool(x) for maxpool in self.maxpools[::-1]]
#         features = torch.cat(features + [x], dim=1)
#     def forward(self, x):
#         feater = [max_pool(x) for max_pool in self.pooling]
#         feater = torch.cat(feater + [x], dim=1)
#         return feater


if __name__ == '__main__':
    S = SpatialPyramidPooling()
    device = torch.device('cuda')
    a = torch.ones(1, 256, 13, 13)
    b = a.clone().detach()
    s = S(b)
    print(s)
