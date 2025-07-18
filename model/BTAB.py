import torch
import torch.nn as nn


class ODConv(nn.Module):
    def __init__(self, in_channels, num_experts=3):
        super(ODConv, self).__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels

        # 初始化多个卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels) for _ in range(num_experts)
        ])
        # 动态权重生成
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        gate_weights = self.gate(x)  # 动态权重
        out = 0
        for i, conv in enumerate(self.convs):
            out += gate_weights[:, i:i+1, :, :] * conv(x)
        return out


def odconv_weights_init(m):
    if isinstance(m, nn.Conv2d):  # 普通 Conv2d
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):  # 全连接层
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, ODConv):  # ODConv 模块
        print(f"Initializing ODConv with {m.num_experts} experts")
        for conv in m.convs:  # 遍历 ODConv 中的所有卷积核
            nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)


class BTAB(nn.Module):
    def __init__(self, in_channels):

        super(BTAB, self).__init__()

        # 背景特征增强模块
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 背景注意力模块（基于膨胀卷积）
        self.bg_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.Sigmoid()  # 生成注意力权重
        )
        # 目标特征增强模块
        self.local_conv = ODConv(in_channels, num_experts=3)
        self.target_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # 点卷积
            nn.Sigmoid()
        )




        # 融合模块
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        # 调用权重初始化函数
        self.apply(odconv_weights_init)

    def forward(self, x):
        # 背景特征处理
        bg_feature = self.global_pool(x)  # 全局池化，生成背景特征
        bg_weight = self.bg_attention(bg_feature)  # 背景注意力
        bg_enhanced = bg_weight * x  # 背景增强特征

        # 目标特征处理
        local_feature = self.local_conv(x)  # 局部特征提取
        target_weight = self.target_attention(local_feature)  # 目标注意力
        target_enhanced = target_weight * x  # 目标增强特征

        # 背景-目标融合
        fused = torch.cat([bg_enhanced, target_enhanced], dim=1)  # 特征拼接
        output = self.fusion(fused)  # 融合特征
        return output

