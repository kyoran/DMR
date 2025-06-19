import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import einops
import math
# from convlstm import *
import numpy as np
import torchvision

from torch.nn.functional import kl_div
from torch.autograd import Variable

from utils.arch_util import *
from utils.Attentions import *

def tie_weights(src, trg):
    assert type(src) == type(trg)
    try:
        trg.weight = src.weight
        trg.bias = src.bias
    except:
        trg = src

def preprocess_obs(rgb_obs, dvs_obs, dvs_obs_shape):
    # print("raw dvs_obs:", torch.isnan(dvs_obs).all(), dvs_obs.min(), dvs_obs.max())

    # RGB
    rgb_obs = rgb_obs / 255.

    # DVS
    if dvs_obs_shape[0] == 5 * 3:   # width * height * 15
        nonzero_ev = (dvs_obs != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = dvs_obs.sum() / num_nonzeros
            stddev = torch.sqrt((dvs_obs ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            if stddev != 0:
                dvs_obs = mask * (dvs_obs - mean) / stddev
        # pass
    elif dvs_obs_shape[0] == 2 * 3:
        dvs_obs = dvs_obs / 255.

    # print("after dvs_obs:", torch.isnan(dvs_obs).all(), dvs_obs.min(), dvs_obs.max())

    return rgb_obs, dvs_obs



class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.obs_shape = obs_shape

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = 6
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs, vis=False):

        if self.obs_shape[0] != 5:
            # ↓↓↓ RGB，DVS-frame，E2VID preprocess
            obs = obs / 255.
            # ↑↑↑

        else:
            # ↓↓↓ DVS-Voxel-grid preprocess！！！
            nonzero_ev = (obs != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                # compute mean and stddev of the **nonzero** elements of the event tensor
                # we do not use PyTorch's default mean() and std() functions since it's faster
                # to compute it by hand than applying those funcs to a masked array
                mean = obs.sum() / num_nonzeros
                stddev = torch.sqrt((obs ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                obs = mask * (obs - mean) / stddev
            # ↑↑↑

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        if vis:
            return conv
        else:
            h = conv.view(conv.size(0), -1)
            return h

    def forward(self, obs, detach=False, vis=False):

        # print("obs.shape:", obs.shape)

        conv = self.forward_conv(obs, vis)

        if detach:
            conv = conv.detach()
        if vis:
            h_fc = self.fc(conv.view(conv.size(0), -1))
        else:
            h_fc = self.fc(conv)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out
        if vis:
            return out, None, conv

        else:
            return out, None

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 4*4  # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.obs_shape = obs_shape

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 6*6  # 1 cameras, input: 128*128
        # out_dims = 14*14  # 1 cameras, input: 256*256

        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()





class pixelCon(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)
        # self.rgb_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)
        # self.dvs_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.con_head_convs = nn.ModuleList()
        self.con_head_convs.append(nn.Conv2d(rgb_obs_shape[0]+dvs_obs_shape[0], 64, 5, stride=2))
        self.con_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.con_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.con_ln = nn.LayerNorm(self.feature_dim)
        # self.con_K = nn.Linear(256 * out_dims, self.feature_dim)
        # self.con_V = nn.Linear(256 * out_dims, self.feature_dim)

        # out_dims = 16  # 1 cameras

        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)


        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        con_conv = torch.relu(self.con_head_convs[0](
            torch.cat([rgb_obs, dvs_obs], dim=1)))
        self.outputs['con_conv1'] = con_conv
        con_conv = torch.relu(self.con_head_convs[1](con_conv))
        self.outputs['con_conv2'] = con_conv
        con_conv = torch.relu(self.con_head_convs[2](con_conv))
        self.outputs['con_conv3'] = con_conv
        con_conv = torch.relu(self.con_head_convs[3](con_conv))
        self.outputs['con_conv4'] = con_conv

        return rgb_conv, dvs_conv, con_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv, con_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()
            con_conv = con_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        con_h = con_conv.view(con_conv.size(0), -1)
        con_h = self.con_ln(self.con_fc(con_h))

        if vis:
            # return torch.cat([rgb_h, con_h, dvs_h], dim=1), [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv]
            return con_h, [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv]
        # return torch.cat([rgb_h, con_h, dvs_h], dim=1), [rgb_h, con_h, dvs_h]
        return con_h, [rgb_h, con_h, dvs_h]


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)



# https://github.com/louislva/deepmind-perceiver/blob/eb668c818eceab957713091ad1e244b14f039f7e/perceiver/positional_encoding.py#L6
# Example parameters: shape=(28, 28), bands=8
# encoding_size = bands*2*2
def positional_encoding(shape, bands):
    # This first "shape" refers to the shape of the input data, not the output of this function
    dims = len(shape)

    # Every tensor we make has shape: (bands, dimension, x, y, etc...)

    # Pos is computed for the second tensor dimension
    # (aptly named "dimension"), with respect to all
    # following tensor-dimensions ("x", "y", "z", etc.)
    pos = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
    )))
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    # Band frequencies are computed for the first
    # tensor-dimension (aptly named "bands") with
    # respect to the index in that dimension
    band_frequencies = (torch.logspace(
        math.log(1.0),
        math.log(shape[0]/2),
        steps=bands,
        base=math.e
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    # For every single value in the tensor, let's compute:
    #             freq[band] * pi * pos[d]

    # We can easily do that because every tensor is the
    # same shape, and repeated in the dimensions where
    # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
    result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

    # Use both sin & cos for each band, and then add raw position as well
    # TODO: raw position
    result = torch.cat([
        torch.sin(result),
        torch.cos(result),
    ], dim=0)

    return result


class DMR_CNN(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))   # input: 9
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)
        # self.rgb_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))  # input: 15
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)
        # self.dvs_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.con_head_convs = nn.ModuleList()
        self.con_head_convs.append(nn.Conv2d(rgb_obs_shape[0]+dvs_obs_shape[0], 64, 5, stride=2))
        self.con_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.con_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.con_ln = nn.LayerNorm(self.feature_dim)
        # self.con_K = nn.Linear(256 * out_dims, self.feature_dim)
        # self.con_V = nn.Linear(256 * out_dims, self.feature_dim)

        # out_dims = 16  # 1 cameras

        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.shape, rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.shape, dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        # ↑↑↑

        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv


        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        con_conv = torch.relu(self.con_head_convs[0](
            torch.cat([rgb_obs, dvs_obs], dim=1)))
        self.outputs['con_conv1'] = con_conv
        con_conv = torch.relu(self.con_head_convs[1](con_conv))
        self.outputs['con_conv2'] = con_conv
        con_conv = torch.relu(self.con_head_convs[2](con_conv))
        self.outputs['con_conv3'] = con_conv
        con_conv = torch.relu(self.con_head_convs[3](con_conv))
        self.outputs['con_conv4'] = con_conv

        return rgb_conv, dvs_conv, con_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv, con_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()
            con_conv = con_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        con_h = con_conv.view(con_conv.size(0), -1)
        con_h = self.con_ln(self.con_fc(con_h))

        if vis:
            # return torch.cat([rgb_h, con_h, dvs_h], dim=1), [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv]
            return con_h, [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv]
        # return torch.cat([rgb_h, con_h, dvs_h], dim=1), [rgb_h, con_h, dvs_h]
        return con_h, [rgb_h, con_h, dvs_h]


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class pixelConV51(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)


        self.con_head_convs = nn.ModuleList()
        self.con_head_convs.append(nn.Conv2d(rgb_obs_shape[0]+dvs_obs_shape[0], 64, 5, stride=2))
        self.con_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.con_head_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.con_fc_rgb = nn.Linear(self.feature_dim, self.feature_dim)
        self.con_fc_dvs = nn.Linear(self.feature_dim, self.feature_dim)
        self.con_fc_com = nn.Linear(self.feature_dim, self.feature_dim)
        self.con_ln_rgb = nn.LayerNorm(self.feature_dim)
        self.con_ln_dvs = nn.LayerNorm(self.feature_dim)
        self.con_ln_com = nn.LayerNorm(self.feature_dim)
        # self.con_K = nn.Linear(256 * out_dims, self.feature_dim)
        # self.con_V = nn.Linear(256 * out_dims, self.feature_dim)

        # self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        # self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)


        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        con_conv = torch.relu(self.con_head_convs[0](
            torch.cat([rgb_obs, dvs_obs], dim=1)))
        self.outputs['con_conv1'] = con_conv
        con_conv = torch.relu(self.con_head_convs[1](con_conv))
        self.outputs['con_conv2'] = con_conv
        con_conv = torch.relu(self.con_head_convs[2](con_conv))
        self.outputs['con_conv3'] = con_conv
        con_conv = torch.relu(self.con_head_convs[3](con_conv))
        self.outputs['con_conv4'] = con_conv

        return rgb_conv, dvs_conv, con_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv, con_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()
            con_conv = con_conv.detach()

        rgb_s = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_s = self.rgb_ln(self.rgb_fc(rgb_s))

        dvs_s = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_s = self.dvs_ln(self.dvs_fc(dvs_s))

        com = con_conv.view(con_conv.size(0), -1)
        com = self.con_head_fc(com)

        rgb_c = self.con_ln_rgb(self.con_fc_rgb(com))
        dvs_c = self.con_ln_dvs(self.con_fc_dvs(com))
        com_c = self.con_ln_com(self.con_fc_com(com))

        # final_z = torch.cat([rgb_c, dvs_c, com_c], dim=1)

        if vis:
            # return final_z, [rgb_s, rgb_c, dvs_s, dvs_c, com_c, com], con_conv, [rgb_conv, dvs_conv]
            return com_c, [rgb_s, rgb_c, dvs_s, dvs_c, com_c, com], con_conv, [rgb_conv, dvs_conv]
        # return final_z, [rgb_s, rgb_c, dvs_s, dvs_c, com_c, com]
        return com_c, [rgb_s, rgb_c, dvs_s, dvs_c, com_c, com]



    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)




class pixelConV41(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)
        # self.rgb_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)
        # self.dvs_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.con_head_convs = nn.ModuleList()
        self.con_head_convs.append(nn.Conv2d(rgb_obs_shape[0]+dvs_obs_shape[0], 64, 5, stride=2))
        self.con_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.con_fc_rgb = nn.Linear(256 * out_dims, self.feature_dim)
        self.con_fc_dvs = nn.Linear(256 * out_dims, self.feature_dim)
        self.con_ln_rgb = nn.LayerNorm(self.feature_dim)
        self.con_ln_dvs = nn.LayerNorm(self.feature_dim)
        # self.con_K = nn.Linear(256 * out_dims, self.feature_dim)
        # self.con_V = nn.Linear(256 * out_dims, self.feature_dim)

        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))

        # out_dims = 16  # 1 cameras

        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)


        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        con_conv = torch.relu(self.con_head_convs[0](
            torch.cat([rgb_obs, dvs_obs], dim=1)))
        self.outputs['con_conv1'] = con_conv
        con_conv = torch.relu(self.con_head_convs[1](con_conv))
        self.outputs['con_conv2'] = con_conv
        con_conv = torch.relu(self.con_head_convs[2](con_conv))
        self.outputs['con_conv3'] = con_conv
        con_conv = torch.relu(self.con_head_convs[3](con_conv))
        self.outputs['con_conv4'] = con_conv

        return rgb_conv, dvs_conv, con_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv, con_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()
            con_conv = con_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        con_h = con_conv.view(con_conv.size(0), -1)
        con_h_rgb = self.con_ln_rgb(self.con_fc_rgb(con_h))
        con_h_dvs = self.con_ln_dvs(self.con_fc_dvs(con_h))

        # con_h = self.alpha * con_h_rgb + (1-self.alpha) * con_h_dvs
        con_h = self.alpha * con_h_rgb + self.beta * con_h_dvs

        if vis:
            # return torch.cat([rgb_h, con_h, dvs_h], dim=1), [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv]
            return con_h, [rgb_h, con_h_rgb, dvs_h, con_h_dvs], con_conv, [rgb_conv, dvs_conv]
        # return torch.cat([rgb_h, con_h, dvs_h], dim=1), [rgb_h, con_h, dvs_h]
        return con_h, [rgb_h, con_h_rgb, dvs_h, con_h_dvs]


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class pixelConNeo(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)
        # self.rgb_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)
        # self.dvs_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)


        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        return rgb_conv, dvs_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        if vis:
            return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h]

        return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h]


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            # tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)

class pixelInputFusion(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras

        self.con_head_convs = nn.ModuleList()
        self.con_head_convs.append(nn.Conv2d(rgb_obs_shape[0]+dvs_obs_shape[0], 64, 5, stride=2))
        self.con_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.con_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        # out_dims = 16  # 1 cameras

        self.outputs = dict()


    def forward_conv(self, obs):
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        con_conv = torch.relu(self.con_head_convs[0](
            torch.cat([rgb_obs, dvs_obs], dim=1)))
        self.outputs['con_conv1'] = con_conv
        con_conv = torch.relu(self.con_head_convs[1](con_conv))
        self.outputs['con_conv2'] = con_conv
        con_conv = torch.relu(self.con_head_convs[2](con_conv))
        self.outputs['con_conv3'] = con_conv
        con_conv = torch.relu(self.con_head_convs[3](con_conv))
        self.outputs['con_conv4'] = con_conv

        return con_conv


    def forward(self, obs, detach=False, vis=False):
        con_conv = self.forward_conv(obs)

        if detach:
            con_conv = con_conv.detach()

        con_h = con_conv.view(con_conv.size(0), -1)
        con_h = self.ln(self.fc(con_h))

        if vis:
            return con_conv

        return con_h, None

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 rgb_vert_anchors, rgb_horz_anchors,
                 dvs_vert_anchors, dvs_horz_anchors,
                 seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, use_velocity=True):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1

        self.rgb_vert_anchors = rgb_vert_anchors
        self.rgb_horz_anchors = rgb_horz_anchors
        self.dvs_vert_anchors = dvs_vert_anchors
        self.dvs_horz_anchors = dvs_horz_anchors

        # positional embedding parameter (learnable), rgb + dvs
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len * rgb_vert_anchors * rgb_horz_anchors +
                                                self.seq_len * dvs_vert_anchors * dvs_horz_anchors,
                                                n_embd))

        # self.pos_emb = nn.Linear(self.seq_len * rgb_vert_anchors * rgb_horz_anchors +
        #                          self.seq_len * dvs_vert_anchors * dvs_horz_anchors,
        #                          n_embd)


        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,    # Mean of the normal distribution with which the linear layers in the GPT are initialized
                                       std=0.02)    # Std  of the normal distribution with which the linear layers in the GPT are initialized
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)   # Initial weight of the layer norms in the gpt.

    def forward(self, rgb_tensor, dvs_tensor):
        """
        Args:
            rgb_tensor (tensor): B*4*seq_len, C, H, W
            dvs_tensor (tensor): B*seq_len, C, H, W
        """
        # print("IIIrgb_tensor:", rgb_tensor.shape)   # ([1, 64, 8, 8])
        # print("IIIdvs_tensor:", dvs_tensor.shape)   # ([1, 64, 8, 8])
        bz = dvs_tensor.shape[0]
        dvs_h, dvs_w = dvs_tensor.shape[2:4]
        rgb_h, rgb_w = rgb_tensor.shape[2:4]

        assert self.seq_len == 1
        rgb_tensor = rgb_tensor.view(bz, self.seq_len, -1, rgb_h, rgb_w).permute(0, 1, 3, 4, 2).contiguous().\
            view(bz, -1, self.n_embd)
        dvs_tensor = dvs_tensor.view(bz, self.seq_len, -1, dvs_h, dvs_w).permute(0, 1, 3, 4, 2).contiguous().\
            view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((rgb_tensor, dvs_tensor), dim=1)


        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        x = x.view(bz,
                   self.seq_len * self.rgb_vert_anchors * self.rgb_horz_anchors +
                   self.seq_len * self.dvs_vert_anchors * self.dvs_horz_anchors,
                   self.n_embd)

        rgb_tensor_out = x[:, :self.seq_len * self.rgb_vert_anchors * self.rgb_horz_anchors, :].contiguous().view(
            bz * self.seq_len, -1, rgb_h, rgb_w)
        dvs_tensor_out = x[:, self.seq_len * self.dvs_vert_anchors * self.dvs_horz_anchors:, :].contiguous().view(
            bz * self.seq_len, -1, dvs_h, dvs_w)

        return rgb_tensor_out, dvs_tensor_out


class pixelCrossFusion(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.avgpool_rgb = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool_dvs = nn.AdaptiveAvgPool2d((8, 8))

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras

        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(3 * 3, 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)

        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(5 * 3, 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)


        # self.transformer1 = GPT(n_embd=64, n_head=2, block_exp=2, n_layer=2,
        #                     rgb_vert_anchors=8, rgb_horz_anchors=8, dvs_vert_anchors=8, dvs_horz_anchors=8,
        #                     seq_len=1, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
        # self.transformer2 = GPT(n_embd=128, n_head=2, block_exp=2, n_layer=2,
        #                     rgb_vert_anchors=8, rgb_horz_anchors=8, dvs_vert_anchors=8, dvs_horz_anchors=8,
        #                     seq_len=1, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
        # self.transformer3 = GPT(n_embd=256, n_head=2, block_exp=2, n_layer=2,
        #                     rgb_vert_anchors=8, rgb_horz_anchors=8, dvs_vert_anchors=8, dvs_horz_anchors=8,
        #                     seq_len=1, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
        self.transformer4 = GPT(n_embd=256, n_head=2, block_exp=2, n_layer=2,
                            rgb_vert_anchors=8, rgb_horz_anchors=8, dvs_vert_anchors=8, dvs_horz_anchors=8,
                            seq_len=1, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)

        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        #############################################################
        # rgb_embd_layer1 = self.avgpool_rgb(rgb_conv)    # ([1, 64, 8, 8])
        # dvs_embd_layer1 = self.avgpool_dvs(dvs_conv)    # ([1, 64, 8, 8])
        # rgb_features_layer1, dvs_features_layer1 = self.transformer1(rgb_embd_layer1, dvs_embd_layer1)
        # rgb_features_layer1 = F.interpolate(rgb_features_layer1,
        #                                       size=(rgb_conv.shape[2], rgb_conv.shape[3]), mode='bilinear',
        #                                       align_corners=False)  # rgb_features_layer1: torch.Size([1, 64, 62, 62])
        # dvs_features_layer1 = F.interpolate(dvs_features_layer1,
        #                                       size=(dvs_conv.shape[2], dvs_conv.shape[3]), mode='bilinear',
        #                                       align_corners=False)  # dvs_features_layer1: torch.Size([1, 64, 62, 62])
        # rgb_conv = rgb_conv + rgb_features_layer1
        # dvs_conv = dvs_conv + dvs_features_layer1
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        #############################################################
        # rgb_embd_layer2 = self.avgpool_rgb(rgb_conv)
        # dvs_embd_layer2 = self.avgpool_dvs(dvs_conv)
        # rgb_features_layer2, dvs_features_layer2 = self.transformer2(rgb_embd_layer2, dvs_embd_layer2)
        # rgb_features_layer2 = F.interpolate(rgb_features_layer2,
        #                                       size=(rgb_conv.shape[2], rgb_conv.shape[3]), mode='bilinear',
        #                                       align_corners=False)
        # dvs_features_layer2 = F.interpolate(dvs_features_layer2,
        #                                       size=(dvs_conv.shape[2], dvs_conv.shape[3]), mode='bilinear',
        #                                       align_corners=False)
        # rgb_conv = rgb_conv + rgb_features_layer2
        # dvs_conv = dvs_conv + dvs_features_layer2
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        #############################################################
        # rgb_embd_layer3 = self.avgpool_rgb(rgb_conv)
        # dvs_embd_layer3 = self.avgpool_dvs(dvs_conv)
        # rgb_features_layer3, dvs_features_layer3 = self.transformer3(rgb_embd_layer3, dvs_embd_layer3)
        # rgb_features_layer3 = F.interpolate(rgb_features_layer3,
        #                                       size=(rgb_conv.shape[2], rgb_conv.shape[3]), mode='bilinear',
        #                                       align_corners=False)
        # dvs_features_layer3 = F.interpolate(dvs_features_layer3,
        #                                       size=(dvs_conv.shape[2], dvs_conv.shape[3]), mode='bilinear',
        #                                       align_corners=False)
        # rgb_conv = rgb_conv + rgb_features_layer3
        # dvs_conv = dvs_conv + dvs_features_layer3
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        #############################################################
        rgb_embd_layer4 = self.avgpool_rgb(rgb_conv)
        dvs_embd_layer4 = self.avgpool_dvs(dvs_conv)
        rgb_features_layer4, dvs_features_layer4 = self.transformer4(rgb_embd_layer4, dvs_embd_layer4)
        rgb_features_layer4 = F.interpolate(rgb_features_layer4,
                                              size=(rgb_conv.shape[2], rgb_conv.shape[3]), mode='bilinear',
                                              align_corners=False)
        dvs_features_layer4 = F.interpolate(dvs_features_layer4,
                                              size=(dvs_conv.shape[2], dvs_conv.shape[3]), mode='bilinear',
                                              align_corners=False)
        rgb_conv = rgb_conv + rgb_features_layer4
        dvs_conv = dvs_conv + dvs_features_layer4
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return rgb_conv, dvs_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))
        if vis:
            return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h], [rgb_conv, dvs_conv]
        else:
            return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h]

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])

        # tie_weights(src=source.transformer1.pos_emb, trg=self.transformer1.pos_emb)
        # tie_weights(src=source.transformer1.ln_f, trg=self.transformer1.ln_f)
        # tie_weights(src=source.transformer2.ln_f, trg=self.transformer2.ln_f)
        # tie_weights(src=source.transformer3.ln_f, trg=self.transformer3.ln_f)
        tie_weights(src=source.transformer4.ln_f, trg=self.transformer4.ln_f)
        for i in range(2):
            # tie_weights(src=source.transformer1.blocks[i], trg=self.transformer1.blocks[i])
            # tie_weights(src=source.transformer2.blocks[i], trg=self.transformer2.blocks[i])
            # tie_weights(src=source.transformer3.blocks[i], trg=self.transformer3.blocks[i])
            tie_weights(src=source.transformer4.blocks[i], trg=self.transformer4.blocks[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class pixelCat(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras

        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(3 * 3, 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)

        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(5 * 3, 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)

        # out_dims = 16  # 1 cameras
        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))

        return rgb_conv, dvs_conv


    def forward(self, obs, detach=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h]

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class TemporalCoderPhase(nn.Module):
    # Transformer-like positional encoding (sin/cos phase) for timestamp
    def __init__(self, feat_size):
        super(TemporalCoderPhase, self).__init__()
        self.ls_by_2 = int(feat_size / 2)
        self.b = torch.arange(0, self.ls_by_2).type("torch.FloatTensor").cuda() # [0, 1, ..., 510, 511]
        self.b_by_latent = self.b / feat_size       # [0/1024, 1/1024, ..., 510/1024, 511/1024]
        self.pow_vec_reci = 1 / torch.pow(1000, (self.b_by_latent)) # [1.0000, 0.9933, ..., 0.0321, 0.0318]

    def forward(self, x, times):
        t = times.view(times.shape[0], -1, 1)

        pes = torch.sin(100 * t * self.pow_vec_reci)
        pec = torch.cos(100 * t * self.pow_vec_reci)

        pe = torch.stack([pes, pec], axis=2).view(
            t.shape[0], t.shape[1], self.ls_by_2 * 2
        )

        # Add embedding to feature data
        return x + pe.permute(0, 2, 1)


class eVAE(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()
        self.feat_size = 256
        self.feature_dim = feature_dim

        self.featnet = nn.ModuleList()
        self.featnet.append(nn.Conv1d(3, 64, 1))
        # self.featnet.append(nn.BatchNorm1d(64))
        self.featnet.append(nn.Conv1d(64, 128, 1))
        # self.featnet.append(nn.BatchNorm1d(128))
        self.featnet.append(nn.Conv1d(128, self.feat_size, 1))
        # self.featnet.append(nn.BatchNorm1d(1024))
        # self.featnet.append(nn.Conv1d(self.feat_size, self.feat_size, 1))


        # self.encoder = nn.ModuleList()
        # self.encoder.append(nn.Linear(self.feat_size, 256))
        # self.encoder.append(nn.Linear(256, self.feature_dim))

        self.fc = nn.Linear(self.feat_size, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.featnet.apply(self.kaiming_init)
        # self.encoder.apply(self.kaiming_init)
        self.temporal_coder = TemporalCoderPhase(self.feat_size)

        self.outputs = dict()


    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    # def sample_latent(self, mu, logvar):
    #     # Reparameterization trick
    #
    #     std = torch.exp(0.5 * logvar)
    #     eps = Variable(std.data.new(std.size()).normal_())
    #
    #     return mu + eps * std


    def extract(self, obs):
        """
        Split event data into spatial and temporal parts and return separately
        """
        # (x, y, p, t)
        # obs.shape: (batch_size, event_num, 4)
        # print("obs.shape:", obs.shape)
        timestamps = obs[:, :, 3].reshape(-1, 1, obs.shape[1])  # # (batch_size, 1, event_num)

        events_no_t = obs[:, :, :3].reshape(-1, obs.shape[1], 3)
        events_no_t = events_no_t.transpose(2, 1).cuda()   # (batch_size, 3, event_num)

        return events_no_t, timestamps


    def forward(self, obs, detach=False):

        outputs = []

        for one_obs in obs:

            # 仅保留多少事件，事件过多会导致显存爆掉
            if one_obs.shape[0] >= 40000:
                # ppp = 0.5  # 保留百分之多少的事件，固定比例
                ppp = 40000 / one_obs.shape[0]  # 保留百分之多少的事件，动态比例
                # 均匀采样
                idx = np.sort(np.random.choice(one_obs.shape[0], int(one_obs.shape[0]*ppp), replace=False))
                one_obs = one_obs[idx]
            # print("@@@one_obs:", one_obs.shape)

            one_obs = one_obs.unsqueeze(0)      # (?, 4)
            x, timestamps = self.extract(one_obs)

            # norm input
            x = x / 128.
            timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)
            # print("0:", x.min(), x.max())                         # 0 - 127
            # print("1:", timestamps.min(), timestamps.max())       # 0 - 1

            # ECN computes per-event spatial features
            for i in range(3):
                # 0, 1
                # 2, 3
                # 4, 5
                x = self.featnet[i](x)
                # x = self.featnet[2 * i](x)
                # x = self.featnet[2 * i + 1](x)
                x = torch.relu(x)
                # print("4:", x.min(), x.max())

            # Temporal embeddings are added to per-event features
            x = torch.relu(self.temporal_coder(x, timestamps))
            # print("1:", x.min(), x.max())
            # print("1:", x.shape)

            # Symmetric function to reduce N features to 1 a la PointNet
            x, _ = torch.max(x, 2, keepdim=True)
            # print("2:", x.shape)

            h = x.view(-1, self.feat_size)

            if detach:
                h = h.detach()
            # print("2:", x.min(), x.max())

            # Compress to latent space
            # h = self.encoder[1](torch.relu(self.encoder[0](x)))
            # mu = dist[:, : self.feature_dim]
            # logvar = dist[:, self.feature_dim:]
            # h = self.sample_latent(mu, logvar)
            # print("3:", h.min(), h.max())



            out = self.ln(self.fc(h))
            outputs.append(out)
            # out = self.fc(h)
            # print("4:", out.shape, out.min(), out.max())

            # return out, [h, mu, logvar]
        outputs = torch.cat(outputs, dim=0)

        return outputs, []


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(len(source.featnet)):
            tie_weights(src=source.featnet[i], trg=self.featnet[i])
        # for i in range(len(source.encoder)):
        #     tie_weights(src=source.encoder[i], trg=self.encoder[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return
        #
        # for k, v in self.outputs.items():
        #     L.log_histogram('train_encoder/%s_hist' % k, v, step)
        #     if len(v.shape) > 2:
        #         L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool =  nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        elif (architecture.startswith('convnext')):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()  # Conv NeXt does not use an activation function after the stem
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()
            _tmp = self._model.global_pool.norm
            self._model.global_pool.norm = nn.LayerNorm((512,1,1), _tmp.eps, _tmp.elementwise_affine)

        _tmp = self._model.conv1
        use_bias = (_tmp.bias != None)
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        # Need to delete the old conv_layer to avoid unused parameters
        del _tmp
        del self._model.stem
        torch.cuda.empty_cache()
        if(use_bias):
            self._model.conv1.bias = _tmp.bias


class pixelHybrid(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # out_dims = 56  # 3 cameras
        out_dims = 36  # 3 cameras

        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)

        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)

        # self.mask_ = DVS_MASK(256)  # RDE_FUSED(256)

        # out_dims = 16  # 1 cameras

        # self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        # self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def forward_conv(self, obs):
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv


        return rgb_conv, dvs_conv

    def forward(self, obs, detach=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)

        # dvs_conv = self.mask_(dvs_conv)

        if detach:
            rgb_conv.detach()
            dvs_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h]

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
        # tie_weights(src=source.mask_.query_conv, trg=self.mask_.query_conv)
        # tie_weights(src=source.mask_.key_conv, trg=self.mask_.key_conv)
        # tie_weights(src=source.mask_.value_conv, trg=self.mask_.value_conv)
        # tie_weights(src=source.mask_.gamma, trg=self.mask_.gamma)


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)

class pixelMultiLevelHybrid(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 4

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # out_dims = 56  # 3 cameras
        out_dims = 36  # 3 cameras

        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_conv_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_conv_ln = nn.LayerNorm(self.feature_dim)

        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_conv_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_conv_ln = nn.LayerNorm(self.feature_dim)

        # 高层
        self.up_fuse_fc1 = nn.Linear(self.feature_dim * 2, 512)
        self.up_fuse_fc2 = nn.Linear(512, self.feature_dim)
        self.up_rgb_fcT = nn.Linear(self.feature_dim, self.feature_dim)
        self.up_rgb_fcT_ln = nn.LayerNorm(self.feature_dim)

        self.up_rgb_fcR = nn.Linear(self.feature_dim, self.feature_dim)
        self.up_rgb_fcR_ln = nn.LayerNorm(self.feature_dim)

        self.up_dvs_fcT = nn.Linear(self.feature_dim, self.feature_dim)
        self.up_dvs_fcT_ln = nn.LayerNorm(self.feature_dim)

        self.up_dvs_fcR = nn.Linear(self.feature_dim, self.feature_dim)
        self.up_dvs_fcR_ln = nn.LayerNorm(self.feature_dim)

        #
        self.down_fuse_fc1 = nn.Linear(self.feature_dim * 2, 512)
        self.down_fuse_fc2 = nn.Linear(512, self.feature_dim)
        self.down_fuse_fc2_ln = nn.LayerNorm(self.feature_dim)

        # out_dims = 16  # 1 cameras

        self.last_fusion_fc = nn.Linear(self.feature_dim * 7, self.feature_dim*2)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim*2)
        self.outputs = dict()

    def forward_conv(self, obs):
        rgb_obs, dvs_obs, _, _ = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        return rgb_conv, dvs_conv


    def intra_modal_extract(self, rgb_conv, dvs_conv):
        # 学习模态内的高层特征
        x = torch.cat([rgb_conv, dvs_conv], dim=1)
        x = self.up_fuse_fc1(x)
        x = torch.relu(x)
        x = self.up_fuse_fc2(x)
        x = torch.relu(x)

        rgb_T = self.up_rgb_fcT(x)
        # rgb_T = self.up_rgb_fcT_ln(rgb_T)

        rgb_R = self.up_rgb_fcR(x)
        # rgb_R = self.up_rgb_fcR_ln(rgb_R)

        dvs_T = self.up_dvs_fcT(x)
        # dvs_T = self.up_dvs_fcT_ln(dvs_T)

        dvs_R = self.up_dvs_fcR(x)
        # dvs_R = self.up_dvs_fcR_ln(dvs_R)

        return rgb_T, rgb_R, dvs_T, dvs_R

    def common_semantic(self, rgb_conv, dvs_conv):
        x = torch.cat([rgb_conv, dvs_conv], dim=1)
        x = self.down_fuse_fc1(x)
        x = torch.relu(x)
        x = self.down_fuse_fc2(x)
        # x = self.down_fuse_fc2_ln(x)

        # x = torch.relu(x)

        return x

    def forward(self, obs, detach=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)
        rgb_conv = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_conv = self.rgb_conv_fc(rgb_conv)
        rgb_conv = self.rgb_conv_ln(rgb_conv)

        dvs_conv = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_conv = self.dvs_conv_fc(dvs_conv)
        dvs_conv = self.dvs_conv_ln(dvs_conv)
        # dvs_conv = self.mask_(dvs_conv)
        ####################################################
        # 学习模态内的高层特征
        rgb_T, rgb_R, dvs_T, dvs_R = self.intra_modal_extract(rgb_conv, dvs_conv)
        ####################################################
        # 学习共有的语义特征
        fuse_D = self.common_semantic(rgb_conv, dvs_conv)
        ####################################################

        if detach:
            rgb_T = rgb_T.detach()
            rgb_R = rgb_R.detach()
            dvs_T = dvs_T.detach()
            dvs_R = dvs_R.detach()
            fuse_D = fuse_D.detach()

        fff = torch.cat([rgb_conv, dvs_conv, rgb_T, rgb_R, dvs_T, dvs_R, fuse_D], dim=1)
        fff = self.last_fusion_fc(fff)
        fff = self.last_fusion_ln(fff)

        return fff, [rgb_conv, dvs_conv, rgb_T, rgb_R, dvs_T, dvs_R, fuse_D]

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])

        tie_weights(src=source.rgb_conv_fc, trg=self.rgb_conv_fc)
        tie_weights(src=source.rgb_conv_ln, trg=self.rgb_conv_ln)

        tie_weights(src=source.dvs_conv_fc, trg=self.dvs_conv_fc)
        tie_weights(src=source.dvs_conv_ln, trg=self.dvs_conv_ln)

        tie_weights(src=source.up_fuse_fc1, trg=self.up_fuse_fc1)
        tie_weights(src=source.up_fuse_fc2, trg=self.up_fuse_fc2)
        tie_weights(src=source.up_rgb_fcT, trg=self.up_rgb_fcT)
        tie_weights(src=source.up_rgb_fcT_ln, trg=self.up_rgb_fcT_ln)
        tie_weights(src=source.up_rgb_fcR, trg=self.up_rgb_fcR)
        tie_weights(src=source.up_rgb_fcR_ln, trg=self.up_rgb_fcR_ln)
        tie_weights(src=source.up_dvs_fcT, trg=self.up_dvs_fcT)
        tie_weights(src=source.up_dvs_fcT_ln, trg=self.up_dvs_fcT_ln)
        tie_weights(src=source.up_dvs_fcR, trg=self.up_dvs_fcR)
        tie_weights(src=source.up_dvs_fcR_ln, trg=self.up_dvs_fcR_ln)

        tie_weights(src=source.down_fuse_fc1, trg=self.down_fuse_fc1)
        tie_weights(src=source.down_fuse_fc2, trg=self.down_fuse_fc2)
        tie_weights(src=source.down_fuse_fc2_ln, trg=self.down_fuse_fc2_ln)



    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # print('@@@1',x.shape)
        # bilinear resizing
        # x = F.upsample(x, (x.size(2)*2-1, x.size(3)*2-1), mode='bilinear')
        x = F.upsample(x, (x.size(2)*2, x.size(3)*2), mode='bilinear', align_corners=True)
        # print('@@@2',x.shape)

        # scaling conv
        x = self.conv2(x)
        # print('@@@3',x.shape)

        return x

class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels//reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels//reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        # print('channel',x.shape)
        return x

class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    Aim: Spatial Attention + Channel Attention
    Output: attention maps with shape identical to input.
    """
    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        # print('spatial_attn',y_spatial.shape)
        y_channel = self.channel_attn(x)
        # print('channel_attn',y_channel.shape)

        y = y_spatial * y_channel
        y = F.sigmoid(self.conv(y))
        return y




class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # print('@@@1',x.shape)
        # bilinear resizing
        # x = F.upsample(x, (x.size(2)*2-1, x.size(3)*2-1), mode='bilinear')
        x = F.upsample(x, (x.size(2)*2, x.size(3)*2), mode='bilinear', align_corners=True)
        # print('@@@2',x.shape)

        # scaling conv
        x = self.conv2(x)
        # print('@@@3',x.shape)

        return x

class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels//reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels//reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        # print('channel',x.shape)
        return x

class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    Aim: Spatial Attention + Channel Attention
    Output: attention maps with shape identical to input.
    """
    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        # print('spatial_attn',y_spatial.shape)
        y_channel = self.channel_attn(x)
        # print('channel_attn',y_channel.shape)

        y = y_spatial * y_channel
        y = F.sigmoid(self.conv(y))
        return y



class ActionMask(nn.Module):
    def __init__(self, channel):
        super(ActionMask, self).__init__()


        cat_channel = channel * 3
        # self.conv1h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv1_rgb = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv1_dvs = nn.Conv2d(channel, channel, kernel_size=3, padding=1)


        self.conv1f_1 = nn.Conv2d(cat_channel, cat_channel, kernel_size=1)
        self.conv1f_2 = nn.Conv2d(cat_channel, channel, kernel_size=1)
        # self.conv1f_2 = nn.Conv2d(cat_channel, channel, kernel_size=7, padding=3)
        # self.conv1f_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        self.conv1f_mask_rgb = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.conv1f_mask_dvs = nn.Conv2d(channel, 1, kernel_size=3, padding=1)


        self.conv3_dvs = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_rgb = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_text = nn.Conv2d(channel, channel, kernel_size=3, padding=1)



    def forward(self, rgb_conv, dvs_conv):
        # if down.size()[2:] != left.size()[2:]:
        #     down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        # out1_rgb = F.relu(self.bn1_rgb(self.conv1_rgb(rgb_conv)), inplace=True)
        # out1_dvs = F.relu(self.bn1_dvs(self.conv1_dvs(dvs_conv)), inplace=True)
        out1_rgb = torch.relu(self.conv1_rgb(rgb_conv))
        out1_dvs = torch.relu(self.conv1_dvs(dvs_conv))
        # print("out1h:", out1_rgb.shape, out1_dvs.shape)
        out1_mul = out1_rgb * out1_dvs     # enhance the common pixels in feature maps, while alleviate the ambiguous ones

        fuse = torch.cat((out1_rgb, out1_dvs), dim=1)   # [1, 128, 62, 62]
        fuse = torch.cat((fuse, out1_mul), 1)           # [1, 192, 62, 62]
        # print("fuse.shape1:", fuse.shape)
        # fuse = nn.AdaptiveAvgPool2d((1, 1))(fuse)   # [1, 192, 1, 1]
        # print("fuse.shape2:", fuse.shape)

        fuse = torch.relu(self.conv1f_1(fuse))      # [1, 192, 1, 1]
        fuse = torch.relu(self.conv1f_2(fuse))

        # out1f = nn.Softmax(dim=1)(self.conv1f(gap)) * gap.shape[1]
        rgb_mask = nn.Sigmoid()(self.conv1f_mask_rgb(fuse))     # [1, 64, 1, 1]
        dvs_mask = nn.Sigmoid()(self.conv1f_mask_dvs(fuse))     # [1, 64, 1, 1]

        out2_rgb = rgb_mask * rgb_conv# + out1_rgb
        out2_dvs = dvs_mask * dvs_conv# + out1_dvs
        texture_fea = (1 - rgb_mask) * rgb_conv + (1 - dvs_mask) * dvs_conv# + fuse


        texture_fea = torch.relu(self.conv3_text(texture_fea))
        rgb_conv = torch.relu(self.conv3_rgb(out2_rgb + out1_rgb))
        dvs_conv = torch.relu(self.conv3_dvs(out2_dvs + out1_dvs))

        return rgb_conv, dvs_conv, texture_fea, rgb_mask, dvs_mask


    # def initialize(self):
    #     weight_init(self)


class PyramidFeatures(nn.Module):
    def __init__(self, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class EventImage_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

class pixelEFNet(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras

        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(3 * 3, 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)

        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(5 * 3, 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)

        self.EICA = EventImage_ChannelAttentionTransformerBlock(256, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')


        self.last_fusion_fc = nn.Linear(256 * 6 * 6, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))

        return rgb_conv, dvs_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)
        fused = self.EICA(rgb_conv, dvs_conv)
        fused_conv = fused.clone()

        if detach:
            fused = fused.detach()

        fused = fused.view(fused.size(0), -1)
        last_h = self.last_fusion_ln(self.last_fusion_fc(fused))

        if vis:
            return last_h, None, fused_conv
        else:
            return last_h, None

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.EICA.norm1_image.body.weight, trg=self.EICA.norm1_image.body.weight)
            tie_weights(src=source.EICA.norm1_event.body.weight, trg=self.EICA.norm1_event.body.weight)
            tie_weights(src=source.EICA.attn.temperature, trg=self.EICA.attn.temperature)
            tie_weights(src=source.EICA.attn.q, trg=self.EICA.attn.q)
            tie_weights(src=source.EICA.attn.k, trg=self.EICA.attn.k)
            tie_weights(src=source.EICA.attn.v, trg=self.EICA.attn.v)
            tie_weights(src=source.EICA.attn.project_out, trg=self.EICA.attn.project_out)

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class pixelFPNNet(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)
        # self.rgb_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)
        # self.dvs_Q = nn.Linear(256 * out_dims, self.feature_dim)

        # FPN
        self.P5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2.4, mode='nearest')
        self.P4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.P3 = nn.Conv2d(256, 256, kernel_size=3, stride=2)

        out_dims = 6 * 6
        self.last_fusion_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)

        rgb_convs, dvs_convs = [], []
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        rgb_convs.append(rgb_conv)
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        rgb_convs.append(rgb_conv)
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        rgb_convs.append(rgb_conv)
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        rgb_convs.append(rgb_conv)
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        dvs_convs.append(dvs_conv)
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        dvs_convs.append(dvs_conv)
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        dvs_convs.append(dvs_conv)
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        dvs_convs.append(dvs_conv)
        self.outputs['dvs_conv4'] = dvs_conv

        return rgb_convs, dvs_convs


    def forward(self, obs, detach=False, vis=False):
        rgb_convs, dvs_convs = self.forward_conv(obs)
        # print("rgb_convs[-1].shape:", rgb_convs[-1].shape)

        P5_x = self.P5(torch.cat((rgb_convs[-1], dvs_convs[-1]), 1))
        # print("P5_x.shape:", P5_x.shape)    # torch.Size([1, 256, 6, 6])
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # print("P5_upsampled_x.shape:", P5_upsampled_x.shape)    # torch.Size([1, 256, 14, 14])

        P4_x = self.P4(torch.cat((rgb_convs[-2], dvs_convs[-2]), 1))
        # print("P4_x.shape:", P4_x.shape)    # torch.Size([1, 256, 6, 6])

        P4_x = P5_upsampled_x + P4_x
        # print("P4_x.shape:", P4_x.shape)

        P3_x = self.P3(P4_x)
        # print("P3_x.shape:", P3_x.shape)

        fused_x = P3_x.clone()
        if detach:
            P3_x = P3_x.detach()


        P3_x = P3_x.view(P3_x.size(0), -1)
        last_h = self.last_fusion_ln(self.last_fusion_fc(P3_x))

        if vis:
            return last_h, None, fused_x

        else:
            return last_h, None


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.P4, trg=self.P4)
            tie_weights(src=source.P5, trg=self.P5)
            # tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)

class pixelRENet(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras


        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(rgb_obs_shape[0], 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)
        # self.rgb_Q = nn.Linear(256 * out_dims, self.feature_dim)


        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(dvs_obs_shape[0], 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)
        # self.dvs_Q = nn.Linear(256 * out_dims, self.feature_dim)

        # RENet
        self.re = REFusion(256, 256, 1)

        out_dims = 6 * 6
        self.last_fusion_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        # rgb_obs, dvs_obs, _, _ = obs
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs
        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        self.outputs['rgb_conv1'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        self.outputs['rgb_conv2'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        self.outputs['rgb_conv3'] = rgb_conv
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))
        self.outputs['rgb_conv4'] = rgb_conv

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        self.outputs['dvs_conv1'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        self.outputs['dvs_conv2'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        self.outputs['dvs_conv3'] = dvs_conv
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))
        self.outputs['dvs_conv4'] = dvs_conv

        return rgb_conv, dvs_conv


    def forward(self, obs, detach=False, vis=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)
        # print("rgb_convs[-1].shape:", rgb_convs[-1].shape)

        fff_conv = self.re(rgb_conv, dvs_conv)  # (256, 6, 6)
        # print("fff_conv:", fff_conv.shape)
        if detach:
            fff_conv = fff_conv.detach()


        fff = fff_conv.view(fff_conv.size(0), -1)
        last_h = self.last_fusion_ln(self.last_fusion_fc(fff))
        if vis:
            return last_h, None, fff_conv
        else:
            return last_h, None


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])
            tie_weights(src=source.re.conv[0], trg=self.re.conv[0])
            tie_weights(src=source.re.conv0_rgb, trg=self.re.conv0_rgb)
            tie_weights(src=source.re.conv1_rgb[0], trg=self.re.conv1_rgb[0])
            tie_weights(src=source.re.conv0_evt, trg=self.re.conv0_evt)
            tie_weights(src=source.re.conv1_evt[0], trg=self.re.conv1_evt[0])
            tie_weights(src=source.re.ChannelGate_rgb.mlp[1], trg=self.re.ChannelGate_rgb.mlp[1])
            tie_weights(src=source.re.ChannelGate_rgb.mlp[3], trg=self.re.ChannelGate_rgb.mlp[3])
            tie_weights(src=source.re.ChannelGate_evt.mlp[1], trg=self.re.ChannelGate_evt.mlp[1])
            tie_weights(src=source.re.ChannelGate_evt.mlp[3], trg=self.re.ChannelGate_evt.mlp[3])
            tie_weights(src=source.re.SpatialGate.spatial.conv, trg=self.re.SpatialGate.spatial.conv,)
            # tie_weights(src=source.con_head_convs[i], trg=self.con_head_convs[i])


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)




_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'pixelCon': pixelCon,
                       'DMR_CNN': DMR_CNN,
                       'pixelConNew': pixelCon,
                       'pixelConNewV2': pixelCon,
                       'pixelConNewV3': pixelCon,
                       'pixelConNewV4': pixelCon,
                       'pixelConNewV4_Repel': pixelCon,
                       'pixelConNewV4_Rec': pixelCon,
                       'pixelCat': pixelCat,
                       'pixelCatSep': pixelCat,
                       'eVAE': eVAE,
                       'pixelHybrid': pixelHybrid,
                       'pixelInputFusion': pixelInputFusion,
                       'pixelMultiLevelHybrid': pixelMultiLevelHybrid,
                       'pixelCrossFusion': pixelCrossFusion,
                       'pixelEFNet': pixelEFNet,
                       'pixelFPNNet': pixelFPNNet,
                       'pixelRENet': pixelRENet,

                       }


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride
    )



if __name__ == '__main__':

    from prettytable import PrettyTable


    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


    count_parameters(DMR_CNN)
