import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# 相对位置嵌入相关函数

def to(x):
    # 返回一个包含设备和数据类型的字典
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    # 如果输入不是元组，则将其转换为元组
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    # 在指定维度上扩展张量
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    # 将相对位置转换为绝对位置
    b, l, m = x.shape
    r = (m + 1) // 2
    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    # 计算一维相对 logits
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2
    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits

class RelPosEmb(nn.Module):
    # 相对位置嵌入模块
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5
        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size
        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')
        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# HaloAttention 类定义了一个基于块和光环的注意力机制

class HaloAttention(nn.Module):
    def __init__(self, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        self.halo_size = halo_size
        inner_dim = dim_head * heads
        self.rel_pos_emb = RelPosEmb(block_size, block_size + (halo_size * 2), dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # 获取块邻域，并准备一个带有光环的版本（带填充的块）以获取键值
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)

        kv_inp = F.unfold(x, kernel_size=block+halo*2, stride=block, padding=halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c=c)

        # 派生查询、键、值
        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # 分割头
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))

        # 缩放
        q *= self.scale

        # 注意力
        sim = einsum('b i d, b j d -> b i j', q, k)

        # 添加相对位置偏置
        sim += self.rel_pos_emb(q)

        # 屏蔽填充（在论文中，他们声称不需要掩码，但是填充呢？）
        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block+(halo*2), stride=block, padding=halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b=b, h=heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # 注意力
        attn = sim.softmax(dim=-1)

        # 聚合
        out = einsum('b i j, b j d -> b i d', attn, v)

        # 合并和组合头
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out = self.to_out(out)

        # 将块合并回原始特征图
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=(h//block), w=(w//block), p1=block, p2=block)
        return out

if __name__ == '__main__':
    # 创建一个随机输入张量
    input = torch.randn(1, 512, 8, 8)
    # 实例化 HaloAttention 模块
    halo = HaloAttention(dim=512, block_size=2, halo_size=1)
    # 通过模型传递输入，获取输出
    output = halo(input)
    # 打印输出张量的形状
    print(output.shape)