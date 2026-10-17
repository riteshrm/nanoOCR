import torch
import torch.nn as nn

class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer="Global",
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([H * W, H + hk - 1, W + wk - 1], dtype="float32")
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask_torch = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2].flatten(
                1
            )
            mask_inf = torch.full([H * W, H * W], "-inf", dtype="float32")
            mask = torch.where(mask_torch < 1, mask_torch, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def forward(self, x):
        qkv = (
            self.qkv(x)
            .reshape((0, -1, 3, self.num_heads, self.head_dim))
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        if self.mixer == "Local":
            attn += self.mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, -1, self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ConvMixer(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        HW=[8, 25],
        local_k=[3, 3],
    ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2D(
            dim,
            dim,
            local_k,
            1,
            [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x

def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = torch.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
    
class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        prenorm=True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == "Global" or mixer == "Local":
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_attr=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=torch.ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr,
        )
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

    
class EncoderWithLightSVTR(nn.Layer):
    """Light SVTR neck: attention with lightweight skip connection.

    Replaces heavy guide fusion with a lightweight 1x1 skip conv + add.
    Adds a DWConv before attention for local context priming.

    When use_guide=True, applies stop_gradient to input so that neck
    gradients do not propagate back to the backbone (implicit regularization).
    """

    def __init__(
        self,
        in_channels,
        dims=64,
        depth=1,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        qk_scale=None,
        local_kernel=7,
        use_guide=False,
    ):
        super().__init__()
        self.use_guide = use_guide
        self.conv_reduce = ConvBNLayer(
            in_channels,
            dims,
            kernel_size=1,
            act=nn.Swish,
        )
        self.local_conv = nn.Sequential(
            nn.Conv2D(
                dims,
                dims,
                [1, local_kernel],
                padding=[0, local_kernel // 2],
                groups=dims,
                bias_attr=False,
                weight_attr=torch.ParamAttr(initializer=KaimingNormal()),
            ),
            nn.BatchNorm2D(dims),
            nn.Swish(),
        )

        self.svtr_block = nn.LayerList(
            [
                Block(
                    dim=dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.Swish,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer="nn.LayerNorm",
                    epsilon=1e-05,
                    prenorm=False,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dims, epsilon=1e-6)

        self.skip_conv = ConvBNLayer(
            in_channels,
            dims,
            kernel_size=1,
            act=nn.Swish,
        )

        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        if self.use_guide:
            x = x.clone()
            x.stop_gradient = True
        skip = self.skip_conv(x)
        z = self.conv_reduce(x)
        z = z + self.local_conv(z)
        B, C, H, W = z.shape
        z = z.flatten(2).transpose([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        z = z.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        z = z + skip
        return z