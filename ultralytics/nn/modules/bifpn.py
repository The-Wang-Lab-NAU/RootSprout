import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN_Add2(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        assert len(in_channels_list) == 2
        # 为每个输入创建独立的 1x1 卷积，统一通道
        self.convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False)
            for in_c in in_channels_list
        ])
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.silu = nn.SiLU()

    def forward(self, x):
        # x: list of 2 tensors
        x = [conv(x_i) for conv, x_i in zip(self.convs, x)]
        # 对齐空间尺寸（以第一个输入为基准）
        target_size = x[0].shape[-2:]  # (H, W)
        for i in range(1, len(x)):
            if x[i].shape[-2:] != target_size:
                x[i] = F.interpolate(x[i], size=target_size, mode='nearest')
        # 加权融合
        weight = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        out = weight[0] * x[0] + weight[1] * x[1]
        return self.silu(out)


class BiFPN_Add3(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        assert len(in_channels_list) == 3
        self.convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False)
            for in_c in in_channels_list
        ])
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.silu = nn.SiLU()

    def forward(self, x):
        x = [conv(x_i) for conv, x_i in zip(self.convs, x)]
        target_size = x[0].shape[-2:]
        for i in range(1, len(x)):
            if x[i].shape[-2:] != target_size:
                x[i] = F.interpolate(x[i], size=target_size, mode='nearest')
        weight = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        out = weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]
        return self.silu(out)