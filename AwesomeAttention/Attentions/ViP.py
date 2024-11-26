import torch
from torch import nn

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) module, which is a simple neural network consisting of multiple linear layers.
    """
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        # The first linear layer maps the input to a hidden representation.
        self.fc1 = nn.Linear(in_features, hidden_features)
        # The activation layer applies a non-linear activation function.
        self.act = act_layer()
        # The second linear layer maps the hidden representation to the output.
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout is a regularization technique to prevent overfitting.
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # The input x is passed through the first linear layer, then the activation function,
        # then dropout, and finally the second linear layer.
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WeightedPermuteMLP(nn.Module):
    """
    A weighted permutation MLP module, which applies different linear transformations to the input along different dimensions,
    then combines them using learned weights.
    """
    def __init__(self, dim, seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim = seg_dim

        # Linear layers for transforming the input along the channel dimension.
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        # Linear layers for transforming the input along the height dimension.
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        # Linear layers for transforming the input along the width dimension.
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        # An MLP for learning the weights for combining the transformed inputs.
        self.reweighting = MLP(dim, dim // 4, dim * 3)

        # A linear layer for the final projection.
        self.proj = nn.Linear(dim, dim)
        # Dropout for the final projection.
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        # Transform the input along the channel dimension.
        c_embed = self.mlp_c(x)

        # Calculate the number of segments.
        S = C // self.seg_dim
        # Transform the input along the height dimension by reshaping and permuting.
        h_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.seg_dim, W, H * S)
        h_embed = self.mlp_h(h_embed).reshape(B, self.seg_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        # Transform the input along the width dimension by reshaping and permuting.
        w_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 1, 2, 4).reshape(B, self.seg_dim, H, W * S)
        w_embed = self.mlp_w(w_embed).reshape(B, self.seg_dim, H, W, S).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        # Calculate the weights for combining the transformed inputs.
        weight = (c_embed + h_embed + w_embed).permute(0, 3, 1, 2).flatten(2).mean(2)
        weight = self.reweighting(weight).reshape(B, C, 3).permute(2, 0, 1).softmax(0).unsqueeze(2).unsqueeze(2)

        # Combine the transformed inputs using the learned weights.
        x = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]

        # Apply the final projection and dropout.
        x = self.proj_drop(self.proj(x))

        return x

if __name__ == '__main__':
    input = torch.randn(64, 8, 8, 512)  # Create a random input tensor.
    seg_dim = 8  # Define the number of segments.
    vip = WeightedPermuteMLP(512, seg_dim)  # Create an instance of WeightedPermuteMLP.
    out = vip(input)  # Pass the input through the module.
    print(out.shape)  # Print the shape of the output tensor.