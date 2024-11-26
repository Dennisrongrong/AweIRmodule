import torch
from torch import nn
from operator import itemgetter
# from axial_attention.reversible import ReversibleSequence
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# 用于保存和设置随机数生成器状态的类，确保模型的可重复性
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    # 记录CPU和GPU的RNG状态
    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    # 根据是否需要记录或设置RNG状态来执行网络
    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# 可逆神经网络块
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx

# 不可逆神经网络块
class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)

# 可逆函数
class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

# 可逆序列
class ReversibleSequence(nn.Module):
    def __init__(self, blocks, ):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim=1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)

# 检查值是否存在
def exists(val):
    return val is not None

# 映射数组元素的索引
def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

# 对数组进行排序并返回索引
def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# 计算轴向注意力的排列
def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]
    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
    return permutations

# 通道层归一化
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# 前归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# 序列
class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

# 排列
class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

# 轴向位置嵌入
class AxialPositionalEmbedding(nn.Module):
    # 轴向位置嵌入类，用于在特定维度上添加位置信息
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []  # 用于存储位置参数
        total_dimensions = len(shape) + 2  # 总维度数
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]  # 排除嵌入维度的其他维度索引

        self.num_axials = len(shape)  # 轴向维度的数量

        # 为每个轴向维度创建位置参数
        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions  # 初始化形状
            shape[emb_dim_index] = dim  # 设置嵌入维度的大小
            shape[axial_dim_index] = axial_dim  # 设置当前轴向维度的大小
            parameter = nn.Parameter(torch.randn(*shape))  # 创建一个随机初始化的参数
            setattr(self, f'param_{i}', parameter)  # 将参数存储为类的属性

    def forward(self, x):
        # 将位置信息添加到输入张量x中
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x

# 注意力机制
class SelfAttention(nn.Module):
    # 自注意力类
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads  # 每个头的维度
        dim_hidden = self.dim_heads * heads  # 隐藏层维度

        self.heads = heads  # 头的数量
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)  # 线性层，将输入转换为查询（Q）
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)  # 线性层，将输入转换为键（K）和值（V）
        self.to_out = nn.Linear(dim_hidden, dim)  # 线性层，将注意力输出转换回原始维度

    def forward(self, x, kv=None):
        kv = x if kv is None else kv  # 如果没有提供kv，则使用x作为kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))  # 分割K和V

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads  # 获取形状信息

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)  # 合并头的函数
        q, k, v = map(merge_heads, (q, k, v))  # 应用函数合并头

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)  # 计算注意力分数
        dots = dots.softmax(dim=-1)  # 应用softmax
        out = torch.einsum('bij,bje->bie', dots, v)  # 计算加权和

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)  # 恢复形状
        out = self.to_out(out)  # 转换回原始维度
        return out

# 轴向注意力类
class AxialAttention(nn.Module):
    # 轴向注意力类，处理多个轴向的注意力
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []  # 存储每个轴向的注意力模块
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        self.axial_attentions = nn.ModuleList(attentions)  # 存储所有轴向的注意力模块
        self.sum_axial_out = sum_axial_out  # 是否对轴向输出求和

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))  # 对所有轴向的输出求和

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)  # 依次应用每个轴向的注意力
        return out

# 轴向图像变换器
class AxialImageTransformer(nn.Module):
    # 轴向图像变换器类
    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)  # 计算排列

        get_ff = lambda: nn.Sequential(  # 获取前馈网络
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding=1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(  # 位置嵌入
            axial_pos_emb_shape) else nn.Identity()

        layers = nn.ModuleList([])  # 存储所有层
        for _ in range(depth):
            attn_functions = nn.ModuleList(  # 注意力层
                [PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in
                 permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])  # 前馈网络层
            layers.append(attn_functions)
            layers.append(conv_functions)

        execute_type = ReversibleSequence if reversible else Sequential  # 选择可逆或不可逆序列
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)  # 应用位置嵌入
        return self.layers(x)  # 应用所有层

# input=torch.randn(3, 128, 7, 7)

# attn = AxialAttention(
#     dim = 3,               # embedding dimension
#     dim_index = 1,         # where is the embedding dimension
#     dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
#     heads = 1,             # number of heads for multi-head attention
#     num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
#     sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
# )
# print(attn(input).shape) # (1, 3, 256, 256)

# 测试代码
if __name__ == '__main__':
    input = torch.randn(3, 128, 7, 7)#(通道数，特征纬度，m*n空间维度)
    model = AxialImageTransformer(  # 创建模型
        dim=128,
        depth=12,
        reversible=True
    )
    outputs = model(input)  # 运行模型
    print(outputs.shape)  # 打印输出形状