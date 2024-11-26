# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.layers import to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple


class LocalAttention(nn.Module):
    # 初始化函数
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()

        # 将窗口大小转换为元组形式，确保其为两个元素的元组
        window_size = to_2tuple(window_size)

        # 线性层，用于将输入特征映射到查询（Q）、键（K）和值（V）
        self.proj_qkv = nn.Linear(dim, 3 * dim)

        # 头的数量
        self.heads = heads

        # 确保维度可以被头的数量整除
        assert dim % heads == 0
        head_dim = dim // heads  # 每个头的维度

        # 缩放因子，用于缩放注意力分数
        self.scale = head_dim ** -0.5

        # 线性层，用于将注意力输出映射回原始维度
        self.proj_out = nn.Linear(dim, dim)

        # 窗口大小
        self.window_size = window_size

        # Dropout层，用于在投影后丢弃部分输出
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        # Dropout层，用于在注意力计算后丢弃部分输出
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        # 初始化相对位置偏置表
        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)  # 使用截断正态分布初始化参数

        # 计算坐标
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='xy'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)  # 注册相对位置索引缓冲区

    def forward(self, x, mask=None):
        # 获取输入数据的维度信息
        B, C, H, W = x.size()
        # 计算每个窗口在高度和宽度方向上的分割数量
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        # 将输入数据x重新排列，以适应窗口分割
        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0],
                                   w1=self.window_size[1])  # B x Nr x Ws x C
        # 进一步重新排列，以适应多头注意力的输入格式
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        # 通过proj_qkv线性层将输入数据映射到查询（Q）、键（K）和值（V）
        qkv = self.proj_qkv(x_total)  # B' x N x 3C
        # 将Q、K、V分开
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # 缩放查询Q
        q = q * self.scale
        # 将Q、K、V重新排列，以适应多头注意力的格式
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        # 计算注意力分数
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        # 获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # 重新排列相对位置偏置，以适应注意力分数的格式
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 将相对位置偏置添加到注意力分数中
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # 当提供掩码时，使用掩码调整注意力分数
            # attn的形状为(b * nW) h w w，mask的形状为nW ww ww
            nW, ww, _ = mask.size()
            # 重新排列attn以匹配mask的形状，并添加掩码
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww,
                                    w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            # 重新排列attn回原来的形状
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        # 应用注意力dropout
        attn = self.attn_drop(attn.softmax(dim=3))

        # 计算加权的值
        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        # 重新排列x以适应线性层的输入
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        # 通过proj_out线性层并将结果通过proj_drop进行dropout
        x = self.proj_drop(self.proj_out(x))  # B' x N x C
        # 将x重新排列回原始的图像形状
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0],
                             w1=self.window_size[1])  # B x C x H x W

        # 返回最终的输出x，以及两个None，代表没有额外的输出
        return x, None, None

class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        # 调用父类的初始化函数
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        # 保存特征图的尺寸和偏移尺寸
        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        # 确保偏移尺寸小于窗口尺寸的最小值
        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        # 创建一个全零的掩码，用于后续生成注意力掩码
        img_mask = torch.zeros(*self.fmap_size)  # H W
        # 定义高度和宽度的切片，用于生成掩码
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                # 为每个窗口分配一个唯一的计数器值
                img_mask[h, w] = cnt
                cnt += 1
        # 重新排列掩码，以适应窗口的格式
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0],w1=self.window_size[1])
        # 生成注意力掩码，通过计算窗口之间的偏移量
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW ww ww
        # 将非零值替换为一个很大的负数，零值保持不变
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # 注册为缓冲区，以便在训练过程中不会更新
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # 对输入x进行偏移操作
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        # 调用父类的前向传播函数，并传入偏移后的输入和注意力掩码
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        # 将经过局部注意力处理的输出再次进行反向偏移，以恢复原始位置
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # 返回最终的输出x，以及两个None，代表没有额外的输出
        return x, None, None


class DAttentionBaseline(nn.Module):
    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, stage_idx
    ):
        super().__init__()
        # 是否使用深度可分离卷积位置编码
        self.dwc_pe = dwc_pe
        # 每个头的通道数
        self.n_head_channels = n_head_channels
        # 缩放因子，用于缩放注意力分数
        self.scale = self.n_head_channels ** -0.5
        # 头的数量
        self.n_heads = n_heads
        # 查询的高和宽
        self.q_h, self.q_w = q_size
        # 键和值的高和宽
        self.kv_h, self.kv_w = kv_size
        # 通道数
        self.nc = n_head_channels * n_heads
        # 分组的数量
        self.n_groups = n_groups
        # 每个分组的通道数
        self.n_group_channels = self.nc // self.n_groups
        # 每个分组的头数
        self.n_group_heads = self.n_heads // self.n_groups
        # 是否使用位置编码
        self.use_pe = use_pe
        # 是否使用固定的位置编码
        self.fixed_pe = fixed_pe
        # 是否不使用偏移
        self.no_off = no_off
        # 偏移范围因子
        self.offset_range_factor = offset_range_factor

        # 根据不同阶段选择不同的核大小
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        # 用于计算偏移量的卷积层
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        # 将输入特征映射到查询、键和值的投影层
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # 将注意力输出映射回原始通道数的投影层
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # 投影后的dropout层
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        # 注意力计算后的dropout层
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            # 如果使用深度可分离卷积位置编码
            if self.dwc_pe:
                # 使用深度可分离卷积来生成相对位置编码
                self.rpe_table = nn.Conv2d(self.nc, self.nc,
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            # 如果使用固定的位置编码
            elif self.fixed_pe:
                # 创建一个参数，用于存储固定的位置编码
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                # 使用截断正态分布初始化位置编码参数
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                # 创建一个参数，用于存储可学习的位置编码
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                # 使用截断正态分布初始化位置编码参数
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            # 如果不使用位置编码，则将rpe_table设置为None
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        # 使用torch.linspace生成高度和宽度方向上的均匀间隔的点
        # 这些点的坐标值从0.5开始，到H_key - 0.5或W_key - 0.5结束，包含端点
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='xy'
        )
        # 将生成的y和x坐标堆叠起来，形成一个包含坐标的张量
        ref = torch.stack((ref_y, ref_x), -1)
        # 将宽度坐标归一化到[0, 2]区间，并转换为相对于参考点的偏移
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        # 将高度坐标归一化到[0, 2]区间，并转换为相对于参考点的偏移
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        # 扩展参考点张量，以匹配输入批次的大小和分组的数量
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):
        # 获取输入数据的维度信息
        B, C, H, W = x.size()
        # 获取输入数据的数据类型和设备
        dtype, device = x.dtype, x.device

        # 通过proj_q投影层将输入特征图x映射到查询（Q）
        q = self.proj_q(x)
        # 重新排列查询Q，以适应分组卷积
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        # 通过conv_offset层计算偏移量
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        # 获取偏移量的高度和宽度
        Hk, Wk = offset.size(2), offset.size(3)
        # 计算样本数量
        n_sample = Hk * Wk

        # 如果设置了偏移范围因子，则对偏移量进行范围限制
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # 重新排列偏移量，以匹配参考点的格式
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # 获取参考点
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # 如果设置了不使用偏移，则将偏移量填充为0
        if self.no_off:
            offset = offset.fill(0.0)

        # 如果设置了偏移范围因子，则根据偏移量调整参考点位置
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            # 如果偏移范围因子小于0，则对偏移量和参考点的和进行tanh操作
            pos = (offset + reference).tanh()

        # 使用grid_sample进行采样
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        # 将采样后的特征图重新排列成原始批次大小和通道数
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # 将原始查询Q重新排列成批次大小和头数
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        # 将采样后的特征图通过proj_k投影层映射到键（K）
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        # 将采样后的特征图通过proj_v投影层映射到值（V）
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        # 使用einsum计算注意力分数
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        # 缩放注意力分数
        attn = attn.mul(self.scale)

        if self.use_pe:
            # 如果使用深度可分离卷积位置编码
            if self.dwc_pe:
                # 通过rpe_table卷积层生成位置编码，并调整形状以匹配查询Q
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            # 如果使用固定的位置编码
            elif self.fixed_pe:
                # 扩展固定位置编码以匹配批次大小
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                # 将位置编码添加到注意力分数中
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                # 如果使用可学习的位置编码
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                # 获取参考点
                q_grid = self._get_ref_points(H, W, B, dtype, device)

                # 计算位移，即参考点和采样点之间的差异
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,n_sample,2).unsqueeze(1)).mul(0.5)

                # 使用grid_sample根据位移采样位置编码
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns
                # 调整形状以匹配注意力分数
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                # 将位置编码添加到注意力分数中
                attn = attn + attn_bias

        # 对注意力分数应用softmax进行归一化
        attn = F.softmax(attn, dim=2)
        # 应用注意力dropout
        attn = self.attn_drop(attn)

        # 使用einsum根据注意力权重和值V计算最终的输出
        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            # 如果使用深度可分离卷积位置编码（DWC_PE），将残差的位置编码加到输出上
            out = out + residual_lepe

        # 将输出重塑回原始的批次大小、通道数、高度和宽度
        out = out.reshape(B, C, H, W)

        # 通过proj_out投影层将输出映射回原始通道数，并通过proj_drop进行dropout
        y = self.proj_drop(self.proj_out(out))

        # 返回最终的输出y，以及偏移量pos和参考点reference的张量
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
class TransformerMLP(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()
        # 输入和输出的维度
        self.dim1 = channels
        self.dim2 = channels * expansion
        # 创建一个顺序容器来存放MLP的各个组件
        self.chunk = nn.Sequential()
        # 添加两个线性层，中间是GELU激活函数和两个dropout层
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):
        # 获取输入特征图的高和宽
        _, _, H, W = x.size()
        # 将输入特征图x重新排列，以适应线性层的输入要求
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        # 通过MLP处理x
        x = self.chunk(x)
        # 将输出重新排列回原始的特征图形状
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 创建一个层归一化模块
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 将输入特征图x重新排列，以适应层归一化的输入要求
        x = einops.rearrange(x, 'b c h w -> b h w c')
        # 通过层归一化处理x
        x = self.norm(x)
        # 将输出重新排列回原始的特征图形状
        return einops.rearrange(x, 'b h w c -> b c h w')
class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()
        # 输入和输出的维度
        self.dim1 = channels
        self.dim2 = channels * expansion
        # 使用1x1卷积来实现MLP的第一个线性层
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        # GELU激活函数
        self.act = nn.GELU()
        # 使用1x1卷积来实现MLP的第二个线性层
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        # 深度可分离卷积，用于在MLP中引入空间信息
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        # 通过深度可分离卷积和第一个线性层处理x
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        # 通过第二个线性层处理x
        x = self.drop2(self.linear2(x))
        return x


class TransformerStage(nn.Module):
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):
        super().__init__()
        fmap_size = to_2tuple(fmap_size)  # 确保特征图尺寸是两个元素的元组
        self.depths = depths  # 该阶段的层数
        hc = dim_embed // heads  # 每个头的维度
        assert dim_embed == heads * hc  # 确保维度匹配
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()  # 输入投影层

        # 创建层归一化模块列表
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        # 创建MLP模块列表
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop)
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        # 创建注意力模块列表
        self.attns = nn.ModuleList()
        # 创建DropPath模块列表
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            # 根据stage_spec指定的类型创建不同的注意力模块
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec={stage_spec[i]} is not supported.')

            # 创建DropPath模块
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)  # 通过输入投影层

        positions = []
        references = []
        for d in range(self.depths):
            x0 = x
            # 通过注意力模块和层归一化
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            # 应用DropPath和残差连接
            x = self.drop_path[d](x) + x0
            x0 = x
            # 通过MLP和层归一化
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references


class DAT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 strides=[-1, -1, -1, -1], offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 **kwargs):
        super().__init__()

        # 根据是否使用卷积补丁投影来创建patch projection层
        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        # 计算图像尺寸
        img_size = img_size // patch_size
        # 计算每个阶段的DropPath比率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 创建Transformer阶段
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                                 dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i],
                                 sr_ratios[i], heads[i], strides[i],
                                 offset_range_factor[i], i,
                                 dwc_pes[i], no_offs[i], fixed_pes[i],
                                 attn_drop_rate, drop_rate, expansion, drop_rate,
                                 dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 use_dwc_mlps[i])
            )
            img_size = img_size // 2

        # 创建下采样投影层
        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        # 创建分类器的归一化和头部
        self.cls_norm = LayerNormProxy(dims[-1])
        self.cls_head = nn.Linear(dims[-1], num_classes)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict):
        # 创建一个新的状态字典用于存储预训练模型的参数
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            # 根据状态字典的键找到模型中的对应模块
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            # 如果预训练模型的参数形状与模型中的参数形状匹配，则添加到新的状态字典中
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # 如果形状不匹配，根据键名处理不同的参数
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # 对于相对位置偏置表，使用双三次插值进行调整
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)

        # 加载新的状态字典到模型中，不严格匹配键名
        self.load_state_dict(new_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        # 通过patch projection层
        x = self.patch_proj(x)
        positions = []
        references = []
        # 遍历每个Transformer阶段
        for i in range(4):
            # 通过当前阶段，获取输出和位置、参考点
            x, pos, ref = self.stages[i](x)
            # 如果不是最后一个阶段，通过下采样投影层
            if i < 3:
                x = self.down_projs[i](x)
            # 保存位置和参考点
            positions.append(pos)
            references.append(ref)
        # 通过分类器的归一化层
        x = self.cls_norm(x)
        # 通过自适应平均池化层，将特征图大小降为1x1
        x = F.adaptive_avg_pool2d(x, 1)
        # 展平特征图
        x = torch.flatten(x, 1)
        # 通过分类器头部
        x = self.cls_head(x)

        # 返回分类结果和位置、参考点
        return x, positions, references

if __name__ == '__main__':
    # 创建一个随机输入张量
    input = torch.randn(1, 3, 224, 224)
    # 创建DAT模型实例
    model = DAT(
        # 省略了部分参数，使用默认值
        img_size=224,
        patch_size=4,
        num_classes=1000,
        expansion=4,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 7, 7],
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    )
    # 通过模型传递输入张量
    output = model(input)
    # 打印分类结果的形状
    print(output[0].shape)