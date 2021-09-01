import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
affine_par = True
from network.resnet import ResNet

BatchNorm2d = nn.BatchNorm2d

class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 value_channels,
                 out_channels=None,
                 scale=1,
                 bn_type=None):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
        )

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 value_channels,
                 out_channels=None,
                 scale=1,
                 bn_type=None):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale, bn_type)


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout=0, sizes=([1]), bn_type=None):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels,
                                                      key_channels, value_channels, size, bn_type) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size, bn_type):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size, bn_type=bn_type)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output

class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=256, dilations=(12, 24, 36), bn_type=None, dropout=0.1):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     BatchNorm2d(out_features),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                              key_channels=out_features//2, value_channels=out_features//2,
                                                              dropout=0, sizes=([2]), bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),

                                   BatchNorm2d(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),

                                   BatchNorm2d(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),

                                   BatchNorm2d(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(out_features*2),
            nn.Dropout2d(dropout)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output

class SFTLayer(nn.Module):
    def __init__(self, inc, outc):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(inc, 256, 1)
        self.SFT_scale_conv1 = nn.Conv2d(256, 256, 1)
        self.SFT_scale_conv2 = nn.Conv2d(256, outc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(inc, 256, 1)
        self.SFT_shift_conv1 = nn.Conv2d(256, 256, 1)
        self.SFT_shift_conv2 = nn.Conv2d(256, outc, 1)

    def forward(self, x, y):
        y = F.upsample(y, [x.size()[2], x.size()[3]], mode='bilinear')
        scale = self.SFT_scale_conv2(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(y), 0.1, inplace=True)))
        shift = self.SFT_shift_conv2(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(y), 0.1, inplace=True)))
        return x * (scale + 1) + shift

class Aquanet(nn.Module):
    def __init__(self, num_classes):
        super(Aquanet, self).__init__()
        self.backbone = ResNet([3, 4, 23, 3])

        self.context1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ASP_OC_Module(256, 256, bn_type=None),
        )
        self.context2 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ASP_OC_Module(256, 256, bn_type=None),
        )
        self.cls1 = nn.Conv2d(512, num_classes-17, kernel_size=1, stride=1, padding=0, bias=True)
        self.cls2 = nn.Conv2d(512, 17, kernel_size=1, stride=1, padding=0, bias=True)

        self.sft1 = SFTLayer(64, 2048)
        self.sft2 = SFTLayer(64, 2048)

        self.sft3 = SFTLayer(17, num_classes-17)
        self.sft4 = SFTLayer(num_classes-17, 17)

        self.dsn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.indices = torch.tensor([0,1,2,3,4,5,39,6,7,8,9,10,40,11,12,41,42,43,44,
                                    45,13,14,15,16,17,18,19,20,21,22,23,46,47,48,49,
                                    50,24,51,25,26,27,28,52,29,53,30,31,32,33,34,35,
                                    36,37,38,54,55])

    def forward(self, x):
        feature_list = self.backbone(x)
        x_l = feature_list[0]
        x = feature_list[-1]
        x_dsn = self.dsn(x)

        x1 = self.sft1(x, x_l[:,:64,:,:])
        x2 = self.sft2(x, x_l[:,64:,:,:])

        x1 = self.context1(x1)
        x2 = self.context2(x2)
        x1 = self.cls1(x1)
        x2 = self.cls2(x2)

        x1_ = self.sft3(x1, x2)
        x2_ = self.sft4(x2, x1)

        x = torch.cat([x1_, x2_], dim=1)

        x = torch.index_select(x, dim=1, index=self.indices.cuda())
        return x_dsn, x
