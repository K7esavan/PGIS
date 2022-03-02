import torch
from torch import nn


class PoseEncoder(nn.module):
    def __init__(self, in_channels):
        super(PoseEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128, 256, "down")
        self.conv8 = conv5x5(256, 256)

    def forward(self, input):
        output = []
        output.append(self.conv1(input))
        output.append(self.conv2(output[-1]))
        output.append(self.conv3(output[-1]))
        output.append(self.conv4(output[-1]))
        output.append(self.conv5(output[-1]))
        output.append(self.conv6(output[-1]))
        output.append(self.conv7(output[-1]))
        output.append(self.conv8(output[-1]))
        return output[-1], [output[5], output[3], output[0]], [output[5], output[3], output[1]]


class ForegroundEncoder(nn.module):
    def __init__(self, in_channels):
        super(PoseEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128, 256, "down")
        self.conv8 = conv5x5(256, 256)

    def forward(self, input):
        output = []
        output.append(self.conv1(input))
        output.append(self.conv2(output[-1]))
        output.append(self.conv3(output[-1]))
        output.append(self.conv4(output[-1]))
        output.append(self.conv5(output[-1]))
        output.append(self.conv6(output[-1]))
        output.append(self.conv7(output[-1]))
        output.append(self.conv8(output[-1]))
        return output[-1], [output[5], output[3], output[0]], [output[5], output[3], output[1]]


class BackgroundEncoder(nn.module):
    def __init__(self, in_channels, input_size, dataset='market1501'):
        super(BackgroundEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32)
        self.conv2 = conv5x5(32, 64, 'down')
        self.di_conv1 = conv3x3(64, 64, rate=2)
        self.di_conv2 = conv3x3(64, 64, rate=4)
        self.di_conv3 = conv3x3(64, 64, rate=8)
        '''
        self.conv3_1 = conv5x5(64, 64)
        '''
        self.conv3 = conv5x5(64, 128, 'down')
        size = (32, 16) if dataset == 'market1501' else (32, 32)
        self.upsample = nn.Upsample(size, mode='bilinear', align_corners=True)

    def forward(self, input):
        output = []
        x = self.conv1(input)
        x = self.conv2(x)
        output.insert(0, x)
        x = self.di_conv1(x)
        x = self.di_conv2(x)
        x = self.di_conv3(x)
        x = self.conv3(x)
        x = self.upsample(x)
        '''
        x = self.conv3_1(x)
        x = self.conv3(x)
        '''
        output.insert(0, x)
        return output


class Decoder(nn.module):
    def __init__(self, in_channels, out_channels, use_nonlocal=True):
        super(Decoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 128, 'up')
        self.conv2 = conv5x5(128 * 3, 128)
        self.conv3 = conv5x5(128, 64, 'up')
        if use_nonlocal:
            self.nonlocal1 = NonlocalBlock(128, 4, False)
            self.nonlocal2 = NonlocalBlock(64, 4, False)
        self.conv4 = conv5x5(64 * 3 + 128, 64)
        self.conv5 = conv5x5(64, 32, 'up')
        self.conv6 = conv5x5(32 * 3 + 64, 32)
        self.conv7 = conv5x5(32, out_channels, "up", sigmoid=True)
        self.use_nonlocal = use_nonlocal

    def forward(self, input, skip, background_feat=None):
        if self.use_nonlocal:
            output = self.conv2(torch.cat([skip[0], self.nonlocal1(self.conv1(input))], 1))
            output = self.conv4(torch.cat([skip[1], self.nonlocal2(self.conv3(output)), background_feat[0]], 1))
        else:
            output = self.conv2(torch.cat([skip[0], self.conv1(input)], 1))
            output = self.conv4(torch.cat([skip[1], self.conv3(output), background_feat[0]], 1))
        output = self.conv6(torch.cat([skip[2], self.conv5(output), background_feat[1]], 1))
        '''
        output = self.conv1(torch.cat([skip[0], self.conv0(input)], 1))
        if self.use_nonlocal:
            output = self.nonlocal1(output)
        output = self.conv3(torch.cat([skip[1], self.conv2(output)], 1))
        if self.use_nonlocal:
            output = self.nonlocal2(output)
        output = self.conv5(torch.cat([skip[2], self.conv4(output), background_feat[0]], 1))
        output = self.conv7(torch.cat([self.conv6(output), background_feat[1]], 1))
        return self.conv8(output)
        '''
        return self.conv7(output)


def conv5x5(in_channels, out_channels, mode=None, sigmoid=False):
    ops = [nn.Conv2d(in_channels, out_channels, 5, padding=2),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(True)]
    if mode == "down":
        ops.insert(0, nn.MaxPool2d(2))
    elif mode == "up":
        ops.insert(0, nn.Upsample(scale_factor=2))
    if sigmoid:
        ops.pop(-1)
        ops.append(nn.Tanh())
    return nn.Sequential(*ops)


def conv3x3(in_channels, out_channels, stride=1, rate=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=rate),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(True))


def conv1x1(in_channels, out_channels, full_seq=True, zero_init=False):
    if full_seq:
        ops = [nn.Conv2d(in_channels, out_channels, 1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(True)]
        if zero_init:
            nn.init.constant_(ops[1].weight, 0.)
        return nn.Sequential(*ops)
    else:
        return nn.Conv2d(in_channels, out_channels, 1)


class NonlocalBlock(nn.Module):
    def __init__(self, in_channels, scale_factor, different_src=False):
        super(NonlocalBlock, self).__init__()
        self.src_attn_conv = conv1x1(in_channels, in_channels // scale_factor, full_seq=False)  # theta
        self.guide_attn_conv = conv1x1(in_channels, in_channels // scale_factor, full_seq=False)  # phi
        self.src_conv = conv1x1(in_channels, in_channels // scale_factor, full_seq=False)  # g
        self.output_conv = conv1x1(in_channels // scale_factor, in_channels, zero_init=True)
        self.different_src = different_src
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        if self.different_src:
            src, guide = input
            batch_size, num_channel, w, h = src.shape
            src_attn = self.src_attn_conv(src).view(batch_size, -1, w * h).permute(0, 2, 1)
            guide_attn = self.guide_attn_conv(guide).view(batch_size, -1, w * h)
            attn = self.softmax(torch.bmm(src_attn, guide_attn))
            src_proj = self.src_conv(src).view(batch_size, num_channel, w * h).permute(0, 2, 1)

            out = torch.bmm(attn, src_proj)
            out = self.output_conv(out) + src
        else:
            batch_size, num_channel, w, h = input.shape
            src_attn = self.src_attn_conv(input).view(batch_size, -1, w * h).permute(0, 2, 1).contiguous()
            guide_attn = self.guide_attn_conv(input).view(batch_size, -1, w * h).contiguous()
            attn = self.softmax(torch.bmm(src_attn, guide_attn))
            src_proj = self.src_conv(input).view(batch_size, -1, w * h).permute(0, 2, 1).contiguous()
            out = torch.bmm(attn, src_proj).permute(0, 2, 1).view(batch_size, -1, w, h).contiguous()
            out = self.output_conv(out) + input

        return out
