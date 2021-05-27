#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
import numpy as np
import conv_cuda


def standardize_weights(w):
    orig_size = w.size()
    w = w.view(w.size(0), -1)

    w_mean = w.mean(1, keepdim=True)
    w = w - w_mean
    w_var = (w * w).mean(1, keepdim=True)

    w = w * torch.rsqrt(w_var.float() + 1e-5).to(dtype=w.dtype)

    return w.view(orig_size)


class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = standardize_weights(self.weight)

        return self._conv_forward(x, weight)


class CheckpointedSeq(nn.Sequential):
    def _forward(self, x):
        for module in self:
            x = module(x)

        return x

    def forward(self, x):
        if x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)


class Dropblock(nn.Module):
    def __init__(self, block_size=3, drop_prob=0.1):
        super().__init__()
        self.drop_prob: float = drop_prob
        self.block_size: int = block_size

    @torch.jit.export
    def set_drop_prob(self, drop_prob: float):
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        n, c, h, w = x.size()
        aspect_ratio = h / w
        bsh = int(aspect_ratio * self.block_size)
        bsw = self.block_size

        gamma = (
            self.drop_prob
            / (bsh * bsw)
            * (h * w)
            / (max(h - bsh + 1, 1) * max(w - bsw + 1, 1))
        )

        mask = torch.rand_like(x[:, 0:1]) < gamma

        if not torch.any(mask):
            return x

        mask = mask.to(dtype=x.dtype)
        block_mask = F.max_pool2d(
            input=mask,
            kernel_size=(bsh, bsw),
            stride=(1, 1),
            padding=(bsh // 2, bsw // 2),
        )

        if block_mask.size(2) > h:
            block_mask = block_mask[..., 0:h, :]

        if block_mask.size(3) > w:
            block_mask = block_mask[..., 0:w]

        block_mask = 1 - block_mask

        scaling = (block_mask.numel() / block_mask.float().sum()).to(dtype=x.dtype)
        return (scaling * block_mask) * x


class BlurPool(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels = channels

        if self.kernel_size == 1:
            a = torch.tensor([1.0,])
        elif self.kernel_size == 2:
            a = torch.tensor([1.0, 1.0])
        elif self.kernel_size == 3:
            a = torch.tensor([1.0, 2.0, 1.0])
        elif self.kernel_size == 4:
            a = torch.tensor([1.0, 3.0, 3.0, 1.0])
        elif self.kernel_size == 5:
            a = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.kernel_size == 6:
            a = torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.kernel_size == 7:
            a = torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        kernel = a.view(-1, 1) * a.view(1, -1)
        kernel = kernel / torch.sum(kernel)
        self.register_buffer(
            "kernel",
            kernel.view(1, 1, kernel_size, kernel_size).repeat(self.channels, 1, 1, 1),
        )

    def forward(self, inp):
        return F.conv2d(
            inp,
            self.kernel,
            stride=self.stride,
            groups=inp.size(1),
            padding=self.padding,
        )


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def gn_relu(ngroups, in_planes, use_normalization=True):
    if use_normalization:
        return [nn.GroupNorm(ngroups, in_planes), nn.ReLU(True)]
    else:
        return [nn.ReLU(True)]


def build_downsample(
    stride,
    inplanes,
    planes,
    expansion,
    ngroups,
    use_normalization,
    use_checkpoint=False,
):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride, padding=1))

        downsample.append(conv1x1(inplanes, planes * expansion))

        if use_normalization:
            downsample.append(nn.GroupNorm(ngroups, planes * expansion))

        if use_checkpoint:
            downsample = CheckpointedSeq(*downsample)
        else:
            downsample = nn.Sequential(*downsample)

    return downsample


class FixupBasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups=1,
        stride=1,
        cardinality=None,
        use_aa=False,
        use_dropblock=False,
        use_checkpoint=False,
    ):
        assert cardinality == 1
        super(FixupBasicBlock, self).__init__()

        self.fixup_bias1a = nn.Parameter(torch.zeros(1))
        self.fixup_bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.fixup_bias2a = nn.Parameter(torch.zeros(1))
        self.fixup_scale = nn.Parameter(torch.ones(1))
        self.fixup_bias2b = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, 1 if use_aa else stride)
        if not use_aa or stride == 1:
            self.aa_pool = nn.Sequential()
        else:
            self.aa_pool = BlurPool(planes, kernel_size=3, stride=stride)

        self.conv2 = conv3x3(planes, planes)
        self.downsample = build_downsample(
            stride, inplanes, planes, self.expansion, ngroups, use_normalization=False,
        )
        if use_dropblock:
            self.dropblock = Dropblock()
        else:
            self.dropblock = None

    def layer_init(self, num_fixups):
        nn.init.normal_(
            self.conv1.weight,
            mean=0,
            std=np.sqrt(
                2 / (self.conv1.weight.shape[0] * np.prod(self.conv1.weight.shape[2:]))
            )
            * num_fixups ** (-0.5),
        )
        nn.init.constant_(self.conv2.weight, 0)
        if self.downsample is not None:
            for l in self.downsample.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(
                        l.weight,
                        mean=0,
                        std=np.sqrt(
                            2 / (l.weight.shape[0] * np.prod(l.weight.shape[2:]))
                        ),
                    )

    def _combine(self, x, identity):
        return torch.relu_(x + identity)

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.fixup_bias1a)
        out = self.relu(out + self.fixup_bias1b)
        if self.dropblock is not None:
            out = self.dropblock(out)

        out = self.aa_pool(out)
        out = self.conv2(out + self.fixup_bias2a)
        out = out * self.fixup_scale + self.fixup_bias2b
        if self.dropblock is not None:
            out = self.dropblock(out)

        if self.downsample is not None:
            identity = self.downsample(x + self.fixup_bias1a)
            if self.dropblock is not None:
                identity = self.dropblock(identity)

        return self._combine(out, identity)


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


def _build_se_branch(planes, r=16):
    return SE(planes, r)


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        cardinality=1,
        use_aa=False,
        use_dropblock=False,
        use_checkpoint=False,
    ):
        super(BasicBlock, self).__init__()

        convs = []

        convs.append(
            conv3x3(
                inplanes, planes, groups=cardinality, stride=1 if use_aa else stride
            )
        )
        convs += gn_relu(ngroups, planes)
        if use_aa and stride != 1:
            convs.append(BlurPool(planes, kernel_size=3, stride=stride))
        convs.append(conv3x3(planes, planes, groups=cardinality))
        convs.append(nn.GroupNorm(ngroups, planes * self.expansion))

        self.convs = nn.Sequential(*convs)

        self.downsample = build_downsample(
            stride, inplanes, planes, self.expansion, ngroups, True
        )

    def _combine(self, x, identity):
        return torch.relu_(x + identity)

    def _impl(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        return self._combine(self.convs(x), identity)

    def forward(self, x):
        return self._impl(x)


class SEFixupBasicBlock(FixupBasicBlock):
    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        cardinality=1,
        use_aa=False,
        use_dropblock=False,
        use_checkpoint=False,
    ):
        super().__init__(
            inplanes,
            planes,
            ngroups,
            stride,
            cardinality,
            use_aa,
            use_dropblock,
            use_checkpoint=use_checkpoint,
        )

        self.se = _build_se_branch(planes * self.expansion, 16)

    def _combine(self, x, identity):
        return torch.relu_(x * self.se(x) + identity)


class SEBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        cardinality=1,
        use_aa=False,
        use_checkpoint=False,
    ):
        super().__init__(
            inplanes,
            planes,
            ngroups,
            stride,
            cardinality,
            use_aa,
            use_checkpoint=use_checkpoint,
        )

        self.se = _build_se_branch(planes * self.expansion, 4)

    def _combine(self, x, identity):
        return torch.relu_(x * self.se(x) + identity)


class SEApply(nn.Module):
    def __init__(self, se):
        super().__init__()
        self.se = se

    def forward(self, x):
        return x * self.se(x)


def _build_bottleneck_branch(
    inplanes,
    planes,
    ngroups,
    stride,
    expansion,
    groups=1,
    se=None,
    use_checkpoint=False,
):
    convs = []
    tmp = []

    tmp.append(conv1x1(inplanes, planes))
    tmp += gn_relu(ngroups, planes)
    if use_checkpoint:
        convs.append(CheckpointedSeq(*tmp))
    else:
        convs += tmp

    tmp = []
    tmp.append(conv3x3(planes, planes, groups=groups))
    tmp += gn_relu(ngroups, planes)
    if use_checkpoint:
        convs.append(CheckpointedSeq(*tmp))
    else:
        convs += tmp

    if se is not None:
        convs.append(SEApply(se))
    if stride != 1:
        convs.append(BlurPool(planes, stride=stride))
    tmp = []
    tmp.append(conv1x1(planes, planes * expansion))
    tmp.append(nn.GroupNorm(ngroups, planes * expansion))
    if use_checkpoint:
        convs.append(CheckpointedSeq(*tmp))
    else:
        convs += tmp

    return nn.Sequential(*convs)


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
        se=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes,
            planes,
            ngroups,
            stride,
            self.expansion,
            groups=cardinality,
            se=se,
            use_checkpoint=use_checkpoint,
        )

        self.downsample = build_downsample(
            stride,
            inplanes,
            planes,
            self.expansion,
            ngroups,
            True,
            use_checkpoint=use_checkpoint,
        )

    def _combine(self, x, identity):
        return torch.relu_(x + identity)

    def _impl(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        return self._combine(self.convs(x), identity)

    def forward(self, x):
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
        use_checkpoint=False,
    ):
        se = _build_se_branch(planes, 8)
        super().__init__(
            inplanes,
            planes,
            ngroups,
            stride,
            downsample,
            cardinality,
            se,
            use_normalization=use_checkpoint,
        )


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


class ResNeXtBottleneck(Bottleneck):
    expansion = 2
    resneXt = True


class SpaceToDepth(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N, C * 16, H // 4, W // 4)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        base_planes,
        ngroups,
        blocks,
        layers,
        cardinality=1,
        use_normalization=True,
        use_checkpoint=False,
    ):
        if isinstance(blocks, list):
            assert len(blocks) == len(layers)
        else:
            blocks = [blocks for _ in layers]

        super().__init__()
        stem = [
            SpaceToDepth(),
            nn.Conv2d(in_channels * 16, base_planes, kernel_size=1, bias=False),
        ]
        stem += gn_relu(ngroups, base_planes, use_normalization)

        self.stem = nn.Sequential(*stem)

        self.cardinality = cardinality

        self.inplanes = base_planes

        self.layers = nn.ModuleList()

        for i in range(len(layers)):
            self.layers.append(
                self._make_layer(
                    blocks[i],
                    ngroups,
                    base_planes * (2 ** i),
                    layers[i],
                    stride=1 if i == 0 and len(layers) == 4 else 2,
                    use_normalization=use_normalization,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)
        self.num_compression_stages = 5

        self.use_normalization = use_normalization

    def _make_layer(
        self,
        block,
        ngroups,
        planes,
        blocks,
        stride=1,
        use_normalization=True,
        use_checkpoint=False,
    ):
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                cardinality=self.cardinality,
                use_checkpoint=use_checkpoint,
            )
        )
        self.inplanes = planes * layers[-1].expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    ngroups,
                    cardinality=self.cardinality,
                    use_checkpoint=use_checkpoint,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        for l in self.layers:
            x = l(x)

        return x

def se_resnet9_fixup(in_channels, base_planes, ngroups):
    return ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEFixupBasicBlock,
        [1, 1, 1, 1],
        use_normalization=False,
    )



class my_se_resnet9_fixup:
    def __init__(self, net, batch_size):
        assert batch_size % 128 == 0
        self.batch_size = batch_size
        self.all_params = []
        self.stem_weight = net.state_dict()["stem.1.weight"]
        self.first_fixup = net.state_dict()["layers.0.0.fixup_bias1a"]

        # I didn't implement downsample (yet)
        self.downsample_params = []
        for i in range(4):
            prefix = "layers.{}.0.".format(i)
            out_img_size = 16//(2**i)
            out_channel = 64 * (2 ** i)
            # probably need to change device if training with multiple GPUs
            # These buffers should be reused if we want to save memory
            ConvOut1 = torch.empty((batch_size, out_img_size, out_img_size, out_channel), dtype=torch.float16, device= torch.device("cuda"))
            ConvOut2 = torch.empty_like(ConvOut1)
            local_params = [net.state_dict()[prefix+"conv1.weight"].permute(0,2,3,1).contiguous(), # My implementation needs NHWC layout. These filters are NCHW
                            ConvOut1,
                            net.state_dict()[prefix+"conv2.weight"].permute(0,2,3,1).contiguous(),
                            ConvOut2,
                            net.state_dict()[prefix+"fixup_bias1b"],
                            net.state_dict()[prefix+"fixup_bias2a"],
                            net.state_dict()[prefix+"fixup_bias2b"],
                            net.state_dict()[prefix+"se.excite.0.weight"],
                            net.state_dict()[prefix+"se.excite.0.bias"],
                            net.state_dict()[prefix+"se.excite.2.weight"],
                            net.state_dict()[prefix+"se.excite.2.bias"],
                            net.state_dict()[prefix+"fixup_scale"]]
            if i != 3:
                local_params = local_params + [net.state_dict()["layers.{}.0.".format(i+1)+"fixup_bias1a"]]
            self.all_params.append(conv_cuda.NetParameters(*local_params))
            if i != 0:
                self.downsample_params.append(net.state_dict()[prefix+"downsample.1.weight"])


    def SpaceToDepth(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N, C * 16, H // 4, W // 4)
        return x
    
    def RunNCHWInput(self, input):
        # both input and stem_weight here are NCHW
        assert input.size(0) == self.batch_size
        NHWC_input = F.conv2d(self.SpaceToDepth(input), self.stem_weight).permute(0, 2, 3, 1).contiguous()
        NHWC_input = F.relu(NHWC_input) + self.first_fixup



        # Modifies self.all_params[0].ConvOut2
        blockIdx = 0
        conv_cuda.Conv2Block1(NHWC_input, self.all_params[blockIdx].Filter1, self.all_params[blockIdx].ConvOut1, self.all_params[blockIdx].Filter2, self.all_params[blockIdx].ConvOut2,
                              self.all_params[blockIdx].fixup_bias1b, self.all_params[blockIdx].fixup_bias2a, self.all_params[blockIdx].fixup_bias2b, self.all_params[blockIdx].fixup_scale)
        # No downsample in first iter
        # Modifies NHWC_input
        conv_cuda.SE1(self.all_params[blockIdx].ConvOut2, self.all_params[blockIdx].SE_W1, self.all_params[blockIdx].SE_b1,
                      self.all_params[blockIdx].SE_W2, self.all_params[blockIdx].SE_b2,  self.all_params[blockIdx].next_fixup_bias1a, NHWC_input)
        


        blockIdx += 1
        out_img_size = 8
        out_channel_size = 128
        in_channel_size = 64
        conv_cuda.Conv2Block2(NHWC_input, self.all_params[blockIdx].Filter1, self.all_params[blockIdx].ConvOut1, self.all_params[blockIdx].Filter2, self.all_params[blockIdx].ConvOut2,
                              self.all_params[blockIdx].fixup_bias1b, self.all_params[blockIdx].fixup_bias2a, self.all_params[blockIdx].fixup_bias2b, self.all_params[blockIdx].fixup_scale)

        ## I should probably implement Avgpool2d in CUDA
        NHWC_input = F.avg_pool2d(NHWC_input.permute(0, 3, 1, 2), 3, stride=2, padding=1).permute(0, 2, 3, 1)

        # 1x1 convolution equivalent to matrix multiplication
        NHWC_input = torch.matmul(NHWC_input.view(self.batch_size * out_img_size * out_img_size, in_channel_size), torch.transpose(
            self.downsample_params[blockIdx-1].view(out_channel_size, in_channel_size), 1, 0)).view(self.batch_size, out_img_size, out_img_size, out_channel_size)


        conv_cuda.SE2(self.all_params[blockIdx].ConvOut2, self.all_params[blockIdx].SE_W1, self.all_params[blockIdx].SE_b1,
                      self.all_params[blockIdx].SE_W2, self.all_params[blockIdx].SE_b2,  self.all_params[blockIdx].next_fixup_bias1a, NHWC_input)




        # block 3
        blockIdx += 1
        out_img_size = 4
        out_channel_size *= 2
        in_channel_size *= 2

        conv_cuda.Conv2Block3(NHWC_input, self.all_params[blockIdx].Filter1, self.all_params[blockIdx].ConvOut1, self.all_params[blockIdx].Filter2, self.all_params[blockIdx].ConvOut2,
                              self.all_params[blockIdx].fixup_bias1b, self.all_params[blockIdx].fixup_bias2a, self.all_params[blockIdx].fixup_bias2b, self.all_params[blockIdx].fixup_scale)

        ## I should probably implement Avgpool2d in CUDA
        NHWC_input = F.avg_pool2d(NHWC_input.permute(0, 3, 1, 2), 3, stride=2, padding=1).permute(0, 2, 3, 1)



        # 1x1 convolution equivalent to matrix multiplication
        NHWC_input = torch.matmul(NHWC_input.view(self.batch_size * out_img_size * out_img_size, in_channel_size), torch.transpose(
            self.downsample_params[blockIdx-1].view(out_channel_size, in_channel_size), 1, 0)).view(self.batch_size, out_img_size, out_img_size, out_channel_size)




        conv_cuda.SE3(self.all_params[blockIdx].ConvOut2, self.all_params[blockIdx].SE_W1, self.all_params[blockIdx].SE_b1,
                      self.all_params[blockIdx].SE_W2, self.all_params[blockIdx].SE_b2,  self.all_params[blockIdx].next_fixup_bias1a, NHWC_input)

        # block 4



        blockIdx += 1
        out_img_size = 2
        out_channel_size *= 2
        in_channel_size *= 2
        conv_cuda.Conv2Block4(NHWC_input, self.all_params[blockIdx].Filter1, self.all_params[blockIdx].ConvOut1, self.all_params[blockIdx].Filter2, self.all_params[blockIdx].ConvOut2,
                              self.all_params[blockIdx].fixup_bias1b, self.all_params[blockIdx].fixup_bias2a, self.all_params[blockIdx].fixup_bias2b, self.all_params[blockIdx].fixup_scale)

        ## I should probably implement Avgpool2d in CUDA
        NHWC_input = F.avg_pool2d(NHWC_input.permute(0, 3, 1, 2), 3, stride=2, padding=1).permute(0, 2, 3, 1)

        # 1x1 convolution equivalent to matrix multiplication
        NHWC_input = torch.matmul(NHWC_input.view(self.batch_size * out_img_size * out_img_size, in_channel_size), torch.transpose(
            self.downsample_params[blockIdx-1].view(out_channel_size, in_channel_size), 1, 0)).view(self.batch_size, out_img_size, out_img_size, out_channel_size)



        conv_cuda.SE4(self.all_params[blockIdx].ConvOut2, self.all_params[blockIdx].SE_W1, self.all_params[blockIdx].SE_b1,
                      self.all_params[blockIdx].SE_W2, self.all_params[blockIdx].SE_b2,  torch.Tensor(), NHWC_input)



        return NHWC_input


