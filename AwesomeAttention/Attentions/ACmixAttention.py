import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入PyTorch库及其神经网络模块和函数模块。


def position(H, W, is_cuda=True):
    # 生成位置编码的函数。
    if is_cuda:
        # 如果使用CUDA，则在GPU上创建位置编码。
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        # 如果不使用CUDA，则在CPU上创建位置编码。
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    # 将水平和垂直位置编码合并为一个四维张量。
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

def stride(x, stride):
    # 对输入张量进行下采样的函数。
    b, c, h, w = x.shape
    # 根据步长对张量进行下采样。
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    # 初始化张量值为0.5的函数。
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    # 初始化张量值为0的函数。
    if tensor is not None:
        tensor.data.fill_(0.)
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        # 初始化模块参数。
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        # 计算每个头的维度。
        self.head_dim = self.out_planes // self.head

        # 定义1x1卷积层，用于生成Q, K, V。
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        # 定义1x1卷积层，用于生成位置编码。
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        # 计算注意力层的填充。
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        # 定义展开层，用于将卷积核展开为二维信号。
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        # 定义softmax层，用于归一化注意力权重。
        self.softmax = torch.nn.Softmax(dim=1)

        # 定义1x1卷积层，用于特征融合。
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        # 定义深度可分离卷积层。
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化网络参数的方法。
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        # 创建深度可分离卷积核。
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        # 前向传播函数。
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        # 计算缩放因子和输出的高度和宽度。
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))
        # 将Q, K, V重塑并应用缩放因子。
        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe
        # 对K和位置编码进行展开。
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,self.kernel_att * self.kernel_att, h_out,w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,w_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)
        # 计算注意力输出。
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

if __name__ == '__main__':
    input = torch.randn(50, 256, 7, 7)
    acmix = ACmix(in_planes=256, out_planes=256)
    output = acmix(input)
    print(output.shape)