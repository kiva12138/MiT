"""
Lightweight CNN (U-Net style) segmentation decoder.

Alternative to DecoderTF. It fuses the infused last-token text feature (reshaped
to a spatial map) with multi-level CLIP feature maps through a few conv blocks.
Selected with --decoder_type CNN.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class DecoderMy(nn.Module):
    def __init__(self, n_classes=2, vision_feature_size=1024):
        super(DecoderMy, self).__init__()

        self.vision_feature_size = vision_feature_size
        hidden_size = 512
        c4_size, c3_size, c2_size, c1_size = 4096, vision_feature_size, vision_feature_size, vision_feature_size
        self.middle_sizes = [64, 128]   # start from 16

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, n_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x_c4, x_c):
        # x_c4: infused text feature [bs, 4096]; x_c: list of CLIP feature maps [bs, 1+hw, vision_feature_size]
        batch_size = x_c4.shape[0]
        feature_size = int((x_c[0].shape[1] - 1) ** 0.5)
        x_c = [x_[:, 1:, :].permute((0, 2, 1)).view(batch_size, self.vision_feature_size, feature_size, feature_size) for x_ in x_c]
        x_c3, x_c2, x_c1 = x_c[0], x_c[1], x_c[2]

        x_c4 = x_c4.view(batch_size, -1, 1, 1)
        x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.relu1_4(self.bn1_4(self.conv1_4(x)))
        x = self.relu2_4(self.bn2_4(self.conv2_4(x)))

        x = F.interpolate(input=x, size=(self.middle_sizes[0], self.middle_sizes[0]), mode='bilinear', align_corners=True)
        x_c2 = F.interpolate(input=x_c2, size=(self.middle_sizes[0], self.middle_sizes[0]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.relu1_3(self.bn1_3(self.conv1_3(x)))
        x = self.relu2_3(self.bn2_3(self.conv2_3(x)))

        x = F.interpolate(input=x, size=(self.middle_sizes[1], self.middle_sizes[1]), mode='bilinear', align_corners=True)
        x_c1 = F.interpolate(input=x_c1, size=(self.middle_sizes[1], self.middle_sizes[1]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))

        x = self.conv1_1(x)
        return x
