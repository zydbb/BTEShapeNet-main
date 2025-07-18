import torch
import torch.nn as nn
import torch.nn.functional as F


class SSB(nn.Module):
    def __init__(self, out_channels):
        """
        MSCE模块支持动态输入通道数
        :param out_channels: 输出通道数
        """
        super(SSB, self).__init__()

        # 多尺度膨胀卷积的参数
        self.out_channels = out_channels

        # 多尺度膨胀卷积分支
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.dilated_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=8, dilation=8)

        # 全局特征融合
        self.global_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 特征融合
        self.fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

        # 自适应加权参数 P
        self.weight_p = nn.Parameter(torch.tensor(0.5))  # 初始值为0.5

    def forward(self, x):
        # 自动获取输入通道数
        in_channels = x.size(1)

        # 多尺度卷积特征提取
        feature1 = self.dilated_conv1(x)
        feature2 = self.dilated_conv2(x)
        feature3 = self.dilated_conv3(x)

        # 全局特征分支
        global_feature = F.adaptive_avg_pool2d(x, 1)
        global_feature = self.global_conv(global_feature)
        global_feature = F.interpolate(global_feature, size=x.size()[2:], mode='bilinear',
                                       align_corners=False)  # 上采样到输入大小

        # 特征拼接和融合
        fused_features = torch.cat([feature1, feature2, feature3,  global_feature], dim=1)
        fused_features = self.fusion_conv(fused_features)

        # 输入特征与融合特征加权相加
        output = self.weight_p * fused_features + (1 - self.weight_p) * x
        return output
