import torch.nn as nn
import math


from .slimmable_ops import USBatchNorm2d
from .slimmable_ops import USConv2d, USLinear, make_divisible
from utils.config import FLAGS


class Block(nn.Module):
    def __init__(self, inp, outp, stride, tmp_ratio=1.0):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = make_divisible(outp // 4)
        expand_ratio = 0.25
        layers = [
            USConv2d(inp, midp, 1, 1, 0, bias=False, ratio=[tmp_ratio, expand_ratio]),
            USBatchNorm2d(midp, ratio=expand_ratio),
            nn.ReLU(inplace=True),

            USConv2d(midp, midp, 3, stride, 1, bias=False, ratio=[expand_ratio, expand_ratio]),
            USBatchNorm2d(midp, ratio=expand_ratio),
            nn.ReLU(inplace=True),

            USConv2d(midp, outp, 1, 1, 0, bias=False, ratio=[expand_ratio, 1]),
            USBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                USConv2d(inp, outp, 1, stride=stride, bias=False, ratio=[tmp_ratio, 1]),
                USBatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        self.features = []
        # head
        assert input_size % 32 == 0

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        init_channel = 64
        channels = make_divisible(init_channel)
        self.block_setting = self.block_setting_dict[FLAGS.depth]
        feats = [64, 128, 256, 512]
        self.features.append(
            nn.Sequential(
                USConv2d(
                    3, channels, 7, 2, 3,
                    bias=False, us=[False, True], ratio=[1, 0.25]),
                USBatchNorm2d(channels, ratio=0.25),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        )

        # body
        for stage_id, n in enumerate(self.block_setting):
            outp = make_divisible(feats[stage_id] * 4)
            for i in range(n):
                if i == 0 and stage_id != 0:
                    self.features.append(Block(channels, outp, 2))
                elif i == 0 and stage_id == 0:
                    self.features.append(Block(channels, outp, 1, tmp_ratio=0.25))
                else:
                    self.features.append(Block(channels, outp, 1))
                channels = outp

        # avg_pool_size = input_size // 32
        # self.features.append(nn.AvgPool2d(avg_pool_size))
        self.features.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.outp = channels
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[True, False])
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
