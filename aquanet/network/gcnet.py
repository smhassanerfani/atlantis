#a
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
affine_par = True
from network.resnet import ResNet

BatchNorm2d = nn.BatchNorm2d

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 one_fc=False):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            if one_fc:
                self.channel_add_conv=nn.Sequential(
                    nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),
                    nn.LayerNorm([self.inplanes, 1, 1]))
            else:
                self.channel_add_conv = nn.Sequential(
                    nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                    nn.LayerNorm([self.planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            if one_fc:
                self.channel_mul_conv=nn.Sequential(
                    nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),
                    nn.LayerNorm([self.inplanes, 1, 1]))
            else:
                self.channel_mul_conv = nn.Sequential(
                    nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                    nn.LayerNorm([self.planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class GCBModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(GCBModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels))
        self.ctb = ContextBlock(inter_channels, ratio=1./4)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        if self.ctb is not None:
            output = self.ctb(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class GCNet(nn.Module):
    def __init__(self, num_classes):
        super(GCNet, self).__init__()
        self.backbone = ResNet([3, 4, 23, 3])

        
        self.head = GCBModule(2048, 512, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        feature_list = self.backbone(x)
        x_dsn = self.dsn(feature_list[-2])
        x = self.head(feature_list[-1])
        return x_dsn, x
