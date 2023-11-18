import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class REFusion(nn.Module):
    def __init__(self, in_planes, out_planes, layer):
        super(REFusion, self).__init__()
        self.ChannelGate_rgb = ChannelGate(out_planes, 16)
        self.ChannelGate_evt = ChannelGate(out_planes, 16)
        self.conv = nn.Sequential(nn.Conv2d(out_planes + out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
        self.SpatialGate = SpatialGate()
        # if (layer != 0) & (layer != 5):
        #     self.projection = nn.Sequential(nn.Conv2d(out_planes // 4, out_planes, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(out_planes, momentum=0.1), nn.ReLU(inplace=True))
        # self.layer = layer
        self.conv0_rgb = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.conv0_evt = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.conv1_rgb = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
        self.conv1_evt = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))

    def forward(self, rgb, evt):
        # if (self.layer != 0) & (self.layer != 5):
        #     evt = self.projection(evt)

        rgb0 = self.conv0_rgb(rgb)
        evt0 = self.conv0_evt(evt)

        mul = rgb0.mul(evt0)

        rgb_mul = rgb0 + mul
        evt_mul = evt0 + mul

        rgb_chn_att = self.ChannelGate_rgb(rgb_mul)
        evt_chn_att = self.ChannelGate_evt(evt_mul)

        rgb_crs_att = rgb_mul * evt_chn_att
        evt_crs_att = evt_mul * rgb_chn_att

        rgb1 = rgb_mul + rgb_crs_att
        evt1 = evt_mul + evt_crs_att

        rgb_spt_att = self.SpatialGate(rgb1)
        evt_spt_att = self.SpatialGate(evt1)

        rgb_crs_att_2 = rgb1 * evt_spt_att
        evt_crs_att_2 = evt1 * rgb_spt_att

        rgb1_2 = rgb1 + rgb_crs_att_2
        evt1_2 = evt1 + evt_crs_att_2

        rgb2 = self.conv1_rgb(rgb1_2)
        evt2 = self.conv1_evt(evt1_2)

        mul2 = torch.mul(rgb2, evt2)

        max_rgb = torch.reshape(rgb2,[rgb2.shape[0],1,rgb2.shape[1],rgb2.shape[2],rgb2.shape[3]])
        max_evt = torch.reshape(evt2,[evt2.shape[0],1,evt2.shape[1],evt2.shape[2],evt2.shape[3]])
        max_cat = torch.cat((max_rgb, max_evt), dim=1)
        max_out = max_cat.max(dim=1)[0]

        out_mul_max = torch.cat((mul2, max_out), dim=1)

        out = out_mul_max

        out = self.conv(out)

        return out