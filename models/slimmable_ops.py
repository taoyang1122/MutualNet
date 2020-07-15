import torch.nn as nn
from utils.config import FLAGS
width_mult = FLAGS.width_mult_range[-1]

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        in_channels_max = in_channels
        out_channels_max = out_channels
        if us[0]:
            in_channels_max = int(make_divisible(
                in_channels
                * width_mult
                / ratio[0]) * ratio[0])
        if us[1]:
            out_channels_max = int(make_divisible(
                out_channels
                * width_mult
                / ratio[1]) * ratio[1])
        groups = in_channels_max if depthwise else 1
        super(USConv2d, self).__init__(
            in_channels_max, out_channels_max,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_basic = in_channels
        self.out_channels_basic = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        in_channels = self.in_channels_basic
        out_channels = self.out_channels_basic
        if self.us[0]:
            in_channels = int(make_divisible(
                self.in_channels_basic
                * self.width_mult
                / self.ratio[0]) * self.ratio[0])
        if self.us[1]:
            out_channels = int(make_divisible(
                self.out_channels_basic
                * self.width_mult
                / self.ratio[1]) * self.ratio[1])
        self.groups = in_channels if self.depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):
            y = y * (max(self.in_channels_list)/self.in_channels)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        in_features_max = in_features
        out_features_max = out_features
        if us[0]:
            in_features_max = make_divisible(
                in_features * width_mult)
        if us[1]:
            out_features_max = make_divisible(
                out_features * width_mult)
        super(USLinear, self).__init__(
            in_features_max, out_features_max, bias=bias)
        self.in_features_basic = in_features
        self.out_features_basic = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        in_features = self.in_features_basic
        out_features = self.out_features_basic
        if self.us[0]:
            in_features = make_divisible(
                self.in_features_basic * self.width_mult)
        if self.us[1]:
            out_features = make_divisible(
                self.out_features_basic * self.width_mult)
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        num_features_max = int(make_divisible(
            num_features * width_mult / ratio) * ratio)
        super(USBatchNorm2d, self).__init__(
            num_features_max, affine=True, track_running_stats=False)
        self.num_features_basic = num_features
        # for tracking log during training
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [
                     int(make_divisible(
                         num_features * width_mult / ratio) * ratio)
                     for width_mult in FLAGS.width_mult_list]
             ]
        )
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = int(make_divisible(
            self.num_features_basic * self.width_mult / self.ratio) * self.ratio)
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y
