import torch
import torch.nn as nn


class IntraCLBlock(nn.Layer):
    def __init__(self, in_channels=96, reduce_factor=4):
        super(IntraCLBlock, self).__init__()
        self.channels = in_channels
        self.rf = reduce_factor
        weight_attr = torch.nn.initializer.KaimingUniform()
        self.conv1x1_reduce_channel = nn.Conv2D(
            self.channels, self.channels // self.rf, kernel_size=1, stride=1, padding=0
        )
        self.conv1x1_return_channel = nn.Conv2D(
            self.channels // self.rf, self.channels, kernel_size=1, stride=1, padding=0
        )

        self.v_layer_7x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(3, 0),
        )
        self.v_layer_5x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.v_layer_3x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
        )

        self.q_layer_1x7 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(0, 3),
        )
        self.q_layer_1x5 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 5),
            stride=(1, 1),
            padding=(0, 2),
        )
        self.q_layer_1x3 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )

        # base
        self.c_layer_7x7 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3),
        )
        self.c_layer_5x5 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )
        self.c_layer_3x3 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.bn = nn.BatchNorm2D(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_new = self.conv1x1_reduce_channel(x)

        x_7_c = self.c_layer_7x7(x_new)
        x_7_v = self.v_layer_7x1(x_new)
        x_7_q = self.q_layer_1x7(x_new)
        x_7 = x_7_c + x_7_v + x_7_q

        x_5_c = self.c_layer_5x5(x_7)
        x_5_v = self.v_layer_5x1(x_7)
        x_5_q = self.q_layer_1x5(x_7)
        x_5 = x_5_c + x_5_v + x_5_q

        x_3_c = self.c_layer_3x3(x_5)
        x_3_v = self.v_layer_3x1(x_5)
        x_3_q = self.q_layer_1x3(x_5)
        x_3 = x_3_c + x_3_v + x_3_q

        x_relation = self.conv1x1_return_channel(x_3)

        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)

        return x + x_relation

class RSELayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        weight_attr = torch.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out

class DilatedReparamBlock(nn.Layer):
    """
    Dilated Reparam Block from UniRepLKNet.
    Reference: https://github.com/AILab-CVC/UniRepLKNet

    Training: uses multiple parallel dilated depthwise convolutions + BN
    Inference: all branches merge into a single large-kernel depthwise conv

    For kernel_size=9, the branches are:
      - origin: 9x9 DW Conv (dil=1)
      - branch1: 5x5 DW Conv (dil=1, equiv RF=5)
      - branch2: 5x5 DW Conv (dil=2, equiv RF=9)
      - branch3: 3x3 DW Conv (dil=3, equiv RF=7)
      - branch4: 3x3 DW Conv (dil=4, equiv RF=9)
    """

    def __init__(self, channels, kernel_size=9, deploy=False):
        super(DilatedReparamBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.is_repped = deploy

        if kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        else:
            raise ValueError(
                "DilatedReparamBlock requires kernel_size in [5,7,9,11,13], "
                "but got {}".format(kernel_size)
            )

        if not self.is_repped:
            self.lk_origin = nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias_attr=False,
            )
            self.origin_bn = nn.BatchNorm2D(channels)

            for k, r in zip(self.kernel_sizes, self.dilates):
                equiv_ks = r * (k - 1) + 1
                p = equiv_ks // 2
                conv = nn.Conv2D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=k,
                    stride=1,
                    padding=p,
                    dilation=r,
                    groups=channels,
                    bias_attr=False,
                )
                bn = nn.BatchNorm2D(channels)
                setattr(self, "dil_conv_k{}_{}".format(k, r), conv)
                setattr(self, "dil_bn_k{}_{}".format(k, r), bn)
        else:
            self.lk_origin = nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias_attr=True,
            )

    def forward(self, x):
        if self.is_repped:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, "dil_conv_k{}_{}".format(k, r))
            bn = getattr(self, "dil_bn_k{}_{}".format(k, r))
            out = out + bn(conv(x))
        return out

    @staticmethod
    def _fuse_bn(conv, bn):
        """Fuse Conv2D + BatchNorm2D into a single Conv2D (weight, bias)."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn._mean
        running_var = bn._variance
        eps = bn._epsilon
        std = torch.sqrt(running_var + eps)
        fused_weight = kernel * (gamma / std).reshape([-1, 1, 1, 1])
        fused_bias = beta - running_mean * gamma / std
        return fused_weight, fused_bias

    @staticmethod
    def _convert_dilated_to_nondilated(kernel, dilate_rate):
        """Convert dilated conv kernel to equivalent non-dilated (sparse) kernel
        by inserting zeros between kernel elements using transposed convolution."""
        if dilate_rate == 1:
            return kernel
        identity = torch.ones(shape=[1, 1, 1, 1], dtype=kernel.dtype)
        # F.conv2d_transpose with stride=dilate_rate inserts zeros
        # Process each channel independently
        C = kernel.shape[0]
        result_list = []
        for i in range(C):
            k_i = kernel[i : i + 1]  # (1, 1, kH, kW)
            dilated = F.conv2d_transpose(k_i, identity, stride=dilate_rate)
            result_list.append(dilated)
        return torch.concat(result_list, axis=0)

    @staticmethod
    def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
        """Pad dilated equivalent kernel to large kernel size and add."""
        large_k = large_kernel.shape[2]
        dilated_k = dilated_kernel.shape[2]
        equiv_ks = dilated_r * (dilated_k - 1) + 1
        equiv_kernel = DilatedReparamBlock._convert_dilated_to_nondilated(
            dilated_kernel, dilated_r
        )
        rows_to_pad = large_k // 2 - equiv_ks // 2
        if rows_to_pad > 0:
            merged = large_kernel + F.pad(
                equiv_kernel, [rows_to_pad, rows_to_pad, rows_to_pad, rows_to_pad]
            )
        else:
            merged = large_kernel + equiv_kernel
        return merged

    @torch.no_grad()
    def rep(self):
        """Merge all parallel branches into a single large-kernel DW conv.
        Call this before switching to deploy/inference mode."""
        if self.is_repped:
            return
        origin_k, origin_b = self._fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, "dil_conv_k{}_{}".format(k, r))
            bn = getattr(self, "dil_bn_k{}_{}".format(k, r))
            branch_k, branch_b = self._fuse_bn(conv, bn)
            origin_k = self._merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b

        merged_conv = nn.Conv2D(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias_attr=True,
        )
        merged_conv.weight.set_value(origin_k)
        merged_conv.bias.set_value(origin_b)
        self.lk_origin = merged_conv
        self.is_repped = True

        delattr(self, "origin_bn")
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, "dil_conv_k{}_{}".format(k, r))
            delattr(self, "dil_bn_k{}_{}".format(k, r))


class MeanPool2D(nn.Layer):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, feat):
        batch_size, channels, _, _ = feat.shape
        feat_flat = torch.reshape(feat, [batch_size, channels, -1])
        feat_mean = torch.mean(feat_flat, axis=2)
        feat_mean = torch.reshape(feat_mean, [batch_size, channels, self.w, self.h])
        return feat_mean
    
class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        if "npu" in torch.device.get_device():
            self.avg_pool = MeanPool2D(1, 1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs

    
class RepLKFPN(nn.Layer):
    """Optimized RSEFPN: replaces 3x3 standard Conv in inp_conv with
    DilatedReparamBlock (DW, 5x5) + PWConv 1x1 + SE.

    Uses the existing DilatedReparamBlock from UniRepLKNet to provide
    multi-branch dilated training with single-conv inference.

    Changes vs RSEFPN:
      - inp_conv: RSELayer(3x3 std Conv + SE)
                → DilatedReparamBlock(5x5 DW) + PWConv(1x1) + SE
      - ins_conv: unchanged (1x1 Conv, no benefit from DW decomposition)

    Parameter comparison (out_channels=96, 4 levels):
      RSEFPN  inp_conv: 4 × (96×24×9 + SE) = 4 × 21,054 = 84,216
      RepLKFPN inp_conv: 4 × (DilReparam96 + 96×24 + SE)
             = 4 × (96×25 + 96×2 + 2×(96×9+96×2) + 96×24 + SE)
             = 4 × (2,400 + 192 + 2×(864+192) + 2,304 + 318)
             = 4 × 7,326 = 29,304
      inp_conv reduction: 65.2%
      Receptive field: 3×3 → 5×5 (with multi-dilation 3,5 patterns in training)

    Inference: DilatedReparamBlock merges to single 5x5 DWConv, zero extra cost.
    """

    def __init__(
        self, in_channels, out_channels, shortcut=True, dilated_kernel_size=7, **kwargs
    ):
        super(RepLKFPN, self).__init__()
        self.out_channels = out_channels
        self.is_repped = False
        self.ins_conv = nn.LayerList()
        self.inp_conv_dw = nn.LayerList()
        self.inp_conv_pw = nn.LayerList()
        self.inp_conv_se = nn.LayerList()
        self.shortcut = shortcut

        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        weight_attr = torch.nn.initializer.KaimingUniform()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            )

            self.inp_conv_dw.append(
                DilatedReparamBlock(
                    channels=out_channels, kernel_size=dilated_kernel_size
                )
            )

            self.inp_conv_pw.append(
                nn.Conv2D(
                    in_channels=out_channels,
                    out_channels=out_channels // 4,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            self.inp_conv_se.append(SEModule(out_channels // 4))

    def _inp_forward(self, x, idx):
        x = self.inp_conv_dw[idx](x)
        x = self.inp_conv_pw[idx](x)
        if self.shortcut:
            x = x + self.inp_conv_se[idx](x)
        else:
            x = self.inp_conv_se[idx](x)
        return x

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        p5 = self._inp_forward(in5, 3)
        p4 = self._inp_forward(out4, 2)
        p3 = self._inp_forward(out3, 1)
        p2 = self._inp_forward(out2, 0)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = torch.concat([p5, p4, p3, p2], axis=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        """Merge DilatedReparamBlock branches for inference deployment."""
        if self.is_repped:
            return
        for i in range(len(self.inp_conv_dw)):
            self.inp_conv_dw[i].rep()
        self.is_repped = True