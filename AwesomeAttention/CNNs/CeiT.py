import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import default_cfgs, _cfg

# 定义导出的模型列表
__all__ = [
    'ceit_tiny_patch16_224', 'ceit_small_patch16_224', 'ceit_base_patch16_224',
    'ceit_tiny_patch16_384', 'ceit_small_patch16_384',
]


# Image2Tokens类，将图像转换为token
class Image2Tokens(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, kernel_size=7, stride=2):
        super(Image2Tokens, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x


# Mlp类，多层感知机
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# LocallyEnhancedFeedForward类，局部增强的前馈网络
class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2, groups=hidden_features)
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        b, n, k = x.size()
        cls_token, tokens = torch.split(x, [1, n - 1], dim=1)
        x = tokens.reshape(b, int(math.sqrt(n - 1)), int(math.sqrt(n - 1)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out


# Attention类，多头自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attention_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# AttentionLCA类，局部类注意力机制
class AttentionLCA(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(AttentionLCA, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.qkv_bias = qkv_bias

    def forward(self, x):
        q_weight = self.qkv.weight[:self.dim, :]
        q_bias = None if not self.qkv_bias else self.qkv.bias[:self.dim]
        kv_weight = self.qkv.weight[self.dim:, :]
        kv_bias = None if not self.qkv_bias else self.qkv.bias[self.dim:]
        B, N, C = x.shape
        _, last_token = torch.split(x, [N - 1, 1], dim=1)
        q = F.linear(last_token, q_weight, q_bias) \
            .reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = F.linear(x, kv_weight, kv_bias) \
            .reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # Transformer中的一个构建块，包含注意力机制和前馈网络
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, with_bn=True,
                 feedforward_type='leff'):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 如果drop_path大于0，则使用DropPath，否则使用Identity
        self.norm2 = norm_layer(dim)  # 第二个LayerNorm
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP的隐藏层维度
        self.norm1 = norm_layer(dim)  # 第一个LayerNorm
        self.feedforward_type = feedforward_type  # 前馈网络类型

        # 根据前馈网络类型选择不同的注意力和前馈网络结构
        if feedforward_type == 'leff':
            self.attn = Attention(  # 使用标准的多头自注意力
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.leff = LocallyEnhancedFeedForward(  # 使用局部增强的前馈网络
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                kernel_size=kernel_size, with_bn=with_bn,
            )
        else:  # 使用局部类注意力（LCA）
            self.attn = AttentionLCA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.feedforward = Mlp(  # 使用标准的MLP
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
            )

    def forward(self, x):
        # 根据前馈网络类型执行不同的操作
        if self.feedforward_type == 'leff':
            x = x + self.drop_path(self.attn(self.norm1(x)))  # 注意力层
            x = x + self.drop_path(self.leff(self.norm2(x)))  # 前馈网络层
            return x, x[:, 0]  # 返回整个序列和第一个token（分类token）
        else:  # LCA
            _, last_token = torch.split(x, [x.size(1)-1, 1], dim=1)  # 分割最后一个token
            x = last_token + self.drop_path(self.attn(self.norm1(x)))  # 注意力层
            x = x + self.drop_path(self.feedforward(self.norm2(x)))  # 前馈网络层
            return x
class HybridEmbed(nn.Module):
    # 混合嵌入层，将CNN特征图嵌入到Transformer中
    def __init__(self, backbone, img_size=224, patch_size=16, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)  # 确保backbone是nn.Module的实例
        img_size = to_2tuple(img_size)  # 将img_size转换为元组
        self.img_size = img_size
        self.backbone = backbone  # CNN骨干网络
        if feature_size is None:  # 如果没有提供feature_size，则自动计算
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # 如果输出是列表或元组，则取最后一个特征图
                feature_size = o.shape[-2:]  # 特征图的尺寸
                feature_dim = o.shape[1]  # 特征图的通道数
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)  # 将feature_size转换为元组
            feature_dim = self.backbone.feature_info.channels()[-1]  # 获取最后一个特征图的通道数
        print('feature_size is {}, feature_dim is {}, patch_size is {}'.format(
            feature_size, feature_dim, patch_size
        ))
        self.num_patches = (feature_size[0] // patch_size) * (feature_size[1] // patch_size)  # 计算patches数量
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)  # 投影层

    def forward(self, x):
        x = self.backbone(x)  # 通过CNN骨干网络
        if isinstance(x, (list, tuple)):
            x = x[-1]  # 如果输出是列表或元组，则取最后一个特征图
        x = self.proj(x).flatten(2).transpose(1, 2)  # 投影并重塑
        return x


class CeIT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 leff_local_size=3,
                 leff_with_bn=True):
        """
        args:
            - img_size (:obj:`int`): input image size
            - patch_size (:obj:`int`): patch size
            - in_chans (:obj:`int`): input channels
            - num_classes (:obj:`int`): number of classes
            - embed_dim (:obj:`int`): embedding dimensions for tokens
            - depth (:obj:`int`): depth of encoder
            - num_heads (:obj:`int`): number of heads in multi-head self-attention
            - mlp_ratio (:obj:`float`): expand ratio in feedforward
            - qkv_bias (:obj:`bool`): whether to add bias for mlp of qkv
            - qk_scale (:obj:`float`): scale ratio for qk, default is head_dim ** -0.5
            - drop_rate (:obj:`float`): dropout rate in feedforward module after linear operation
                and projection drop rate in attention
            - attn_drop_rate (:obj:`float`): dropout rate for attention
            - drop_path_rate (:obj:`float`): drop_path rate after attention
            - hybrid_backbone (:obj:`nn.Module`): backbone e.g. resnet
            - norm_layer (:obj:`nn.Module`): normalization type
            - leff_local_size (:obj:`int`): kernel size in LocallyEnhancedFeedForward
            - leff_with_bn (:obj:`bool`): whether add bn in LocallyEnhancedFeedForward
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # 使用混合嵌入层将CNN特征图嵌入到Transformer中
        self.i2t = HybridEmbed(
            hybrid_backbone, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.i2t.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # 定义每个block的drop path概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                kernel_size=leff_local_size, with_bn=leff_with_bn)
            for i in range(depth)])

        # without droppath， 没有droppath的block
        self.lca = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
            feedforward_type = 'lca'
        )
        self.pos_layer_embed = nn.Parameter(torch.zeros(1, depth, embed_dim))
        # 最后的LayerNorm层
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # 初始化位置嵌入和类别token的权重
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    # 权重初始化函数
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    # 定义不需要权重衰减的参数
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.i2t(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 注意力在cls tokens上
        cls_token_list = []
        for blk in self.blocks:
            x, curr_cls_token = blk(x)
            cls_token_list.append(curr_cls_token)

        all_cls_token = torch.stack(cls_token_list, dim=1)  # B*D*K
        all_cls_token = all_cls_token + self.pos_layer_embed
        # attention over cls tokens
        last_cls_token = self.lca(all_cls_token)
        last_cls_token = self.norm(last_cls_token)

        return last_cls_token.view(B, -1)

    # 前向传播函数，进行分类
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


from timm.models import register_model
from .cait import CeIT  # 假设CeiT类定义在cait模块中
from .image2tokens import Image2Tokens  # 假设Image2Tokens类定义在image2tokens模块中


# 使用装饰器register_model注册模型，这样模型就可以通过名称被检索和实例化
@register_model
def ceit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    创建一个具有以下特点的CeiT模型：
    - 卷积 + 池化作为特征提取的干（stem）
    - 局部增强的前馈网络（Locally Enhanced FeedForward）
    - 对类别token进行注意力操作
    """
    # 实例化Image2Tokens类，用于将输入图像转换为token
    i2t = Image2Tokens()

    # 创建CeiT模型实例，配置为“tiny”大小
    model = CeIT(
        hybrid_backbone=i2t,  # 使用Image2Tokens作为混合骨干网络
        patch_size=4,  # patch尺寸为4
        embed_dim=192,  # 嵌入维度为192
        depth=12,  # 模型深度为12
        num_heads=3,  # 多头自注意力的头数为3
        mlp_ratio=4,  # MLP的扩展比率为4
        qkv_bias=True,  # 在QKV中添加偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用LayerNorm，epsilon值为1e-6
        **kwargs  # 其他关键字参数
    )

    # 设置模型的默认配置
    model.default_cfg = _cfg()

    # 如果需要预训练权重，则加载预训练权重
    if pretrained:
        # 这里省略了加载预训练权重的代码，因为具体的URL和加载逻辑未给出
        pass

    # 返回创建的模型实例
    return model
#后续同理

@register_model
def ceit_small_patch16_224(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t,
        patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def ceit_base_patch16_224(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t,
        patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def ceit_tiny_patch16_384(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t, img_size=384,
        patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def ceit_small_patch16_384(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t, img_size=384,
        patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CeIT(
        hybrid_backbone=Image2Tokens(),
        patch_size=4, 
        embed_dim=192, 
        depth=12, 
        num_heads=3, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    output=model(input)
    print(output.shape)