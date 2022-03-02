import torch
from torch import nn, optim
from torchvision.ops import roi_pool

import mrnet
import vgg


class Model(nn.Module):
    def __init__(self, pose_channels=3, img_channels=3, input_size=128,
                 recurrent=1, use_nonlocal=True, dataset=None, lr=2e-4,
                 lambda1=0.5, lambda2=0.05, lambda3=0.5, mapping=None, roi_size=None):
        super(Model, self).__init__()

        self.model = Pose_to_Image(pose_channels, img_channels, input_size, recurrent,
                                   use_nonlocal, dataset)

        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.schd = optim.lr_scheduler.StepLR(self.optim, 50, 0.5)

        # for perceptual loss
        with torch.no_grad():
            self.vgg_layers = vgg.vgg19(pretrained=True).features

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.recurrent = recurrent
        self.layers_mapping = mapping
        self.roi_size = roi_size

    def scheduler_step(self):
        self.schd.step()
        if self.use_gan:
            self.schd_d_pose.step()
            self.schd_d_img.step()

    def forward(self, input):
        self.input = input
        outputs = self.model(input)
        return outputs

    def cal_perc_feat(self, x, target_bbox=None):
        initial_size = x.size()
        image_w = initial_size[2]
        output = {}
        mask_output = {}
        roi_cnt = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layers_mapping:
                if target_bbox is not None:
                    spatial_scale = x.shape[2] / image_w
                    mask_output[self.layers_mapping[name]] = roi_pool(x, target_bbox,
                                                                      self.roi_size[roi_cnt], spatial_scale)
                    roi_cnt += 1
                output[self.layers_mapping[name]] = x
        return output, mask_output

    def cal_perc_loss(self, pred, target, roi_bbox=None):
        output = []
        perc_loss = []
        mask_perc_loss = []
        bs = pred.shape[0]
        feat, mask_feat = self.cal_perc_feat(torch.cat([pred, target], 0), roi_bbox)
        for k in feat.keys():
            perc_loss.append(torch.mean((feat[k][:bs] - feat[k][bs:]).pow(2)))
            if roi_bbox is not None:
                mask_perc_loss.append(torch.mean((mask_feat[k][:bs] - mask_feat[k][bs:]).pow(2)))
        perc_loss = torch.mean(torch.stack(perc_loss))
        mask_perc_loss = torch.mean(torch.stack(mask_perc_loss)) if roi_bbox is not None else None
        return perc_loss, mask_perc_loss

    def backward_g(self, pred, input, target, roi_bbox=None):
        self.optim.zero_grad()

        final_pred = pred[-1]
        pose_imgs, human_imgs, background_imgs, masks = input
        src_imgs = human_imgs + background_imgs
        perc_loss = []
        mask_perc_loss = []
        l1_loss = []
        for i in range(self.recurrent):
            pred_ = pred[i]
            l1_loss.append(nn.L1Loss()(pred_, target))
            # l1_loss.append(nn.MSELoss()(pred_, target))
            perc_loss_, mask_perc_loss_ = self.cal_perc_loss(pred_, target, roi_bbox)
            perc_loss.append(perc_loss_)
            if roi_bbox is not None:
                mask_perc_loss.append(mask_perc_loss_)

        l1_loss = torch.stack(l1_loss, 0).mean()
        perc_loss = torch.stack(perc_loss, 0).mean()

        g_loss = l1_loss + self.lambda1 * perc_loss
        # g_loss = self.lambda1 * (l1_loss + perc_loss)
        if len(mask_perc_loss) > 0:
            mask_perc_loss = torch.stack(mask_perc_loss, 0).mean()
            g_loss = g_loss + self.lambda2 * mask_perc_loss
        else:
            mask_perc_loss = None

        gan_loss = None

        g_loss.backward()
        self.optim.step()

        return l1_loss, perc_loss, mask_perc_loss, gan_loss

    def backward_d_p(self, net_d, fake_input, real_input, optim_d):
        optim_d.zero_grad()
        d_loss = self.lambda3 * (self.crit_gan(net_d(real_input), True) + \
                                 self.crit_gan(net_d(fake_input), False))
        d_loss.backward()
        optim_d.step()
        return d_loss

    def backward_d(self, pred, input, target):
        pose_imgs, human_imgs, background_imgs, masks = input
        src_imgs = human_imgs + background_imgs

        fake_pose_input = torch.cat([pose_imgs, pred[-1]], 1).detach()
        real_pose_input = torch.cat([pose_imgs, target], 1).detach()

        fake_img_input = torch.cat([src_imgs, pred[-1]], 1).detach()
        real_img_input = torch.cat([src_imgs, target], 1).detach()

        d_pose_loss = self.backward_d_p(self.net_d_pose, fake_pose_input, real_pose_input, self.optim_d_pose)
        d_img_loss = self.backward_d_p(self.net_d_img, fake_img_input, real_img_input, self.optim_d_img)

        d_loss = d_pose_loss + d_img_loss
        return d_loss

    def optimize(self, pose_imgs, human_imgs, background_imgs, masks, target_imgs, roi_bbox):
        input = (pose_imgs, human_imgs, background_imgs, masks)
        pred = self.forward(input)

        l1_loss, perc_loss, mask_perc_loss, gan_loss = self.backward_g(pred, input, target_imgs, roi_bbox)

        d_loss = None

        return pred, l1_loss, perc_loss, mask_perc_loss, gan_loss, d_loss


class Pose_to_Image(nn.Module):
    def __init__(self, pose_channels=3, img_channels=3, input_size=128,
                 recurrent=1, use_nonlocal=True, dataset=None):
        super(Pose_to_Image, self).__init__()
        self.pose_channels = pose_channels
        self.img_channels = img_channels
        self.encoder_conv = MainEncoder(pose_channels, img_channels)
        self.decoder_conv = mrnet.Decoder(256, 3, use_nonlocal=use_nonlocal)
        self.background_encoder = mrnet.BackgroundEncoder(img_channels + 1, input_size, dataset=dataset)
        self.recurrent = recurrent

    def forward(self, input):
        pose_imgs, human_imgs, background_imgs, masks = input
        foreground_feat, skip = self.encoder_conv(pose_imgs, human_imgs)
        outputs = []
        for i in range(self.recurrent):
            background_feat = self.background_encoder(torch.cat([background_imgs, masks], 1))
            output = self.decoder_conv(foreground_feat, skip, background_feat)
            background_imgs = output
            outputs.append(output)
        return outputs


class MainEncoder(nn.Module):
    def __init__(self, pose_channels, foreground_channels):
        super(MainEncoder, self).__init__()
        self.pose_enc = mrnet.PoseEncoder(pose_channels)
        self.fore_enc = mrnet.ForegroundEncoder(foreground_channels)
        # self.conv = conv1x1(512, 256)
        self.conv = BasicBlock(512, 256, downsample=nn.Sequential(
            mrnet.conv1x1(512, 256),
            nn.BatchNorm2d(256),
        ))

    def forward(self, pose, foreground):
        pose_feat, lateral, pose_skip = self.pose_enc(pose)
        fore_feat, fore_skip = self.fore_enc(foreground, lateral)
        skip = list(map(lambda x: torch.cat(list(x), 1), zip(pose_skip, fore_skip)))
        feat = torch.cat([pose_feat, fore_feat], 1)
        feat = self.conv(feat)
        return feat, skip


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
