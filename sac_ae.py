import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

import utils
from encoder import make_encoder
from torch.distributions import Categorical

LOG_FREQ = 10000

def get_dim(encoder_type, feature_dim):
    if encoder_type == "pixelHybrid":
        fff_dim = feature_dim * 2
    elif encoder_type == "pixelHybridActionMask":
        fff_dim = feature_dim * 3
    elif encoder_type == "pixelHybridActionMaskV2":
        fff_dim = feature_dim * 3
    elif encoder_type == "pixelHybridActionMaskV3":
        fff_dim = feature_dim
    elif encoder_type == "pixelHybridActionMaskV4":
        fff_dim = feature_dim
    elif encoder_type == "pixelHybridActionMaskV5":
        fff_dim = feature_dim
    elif encoder_type == "pixelMultiLevelHybrid":
        fff_dim = feature_dim*2
    elif encoder_type == "pixelWAE":
        fff_dim = feature_dim*1
    elif encoder_type == "pixelConNeo":
        fff_dim = feature_dim * 2
    elif encoder_type == "pixelCon":
        fff_dim = feature_dim * 1
    elif encoder_type == "pixelConV51":
        fff_dim = feature_dim * 1
    elif encoder_type == "pixelCat" or encoder_type == "pixelCatSep" or encoder_type == "pixelCrossFusion":
        fff_dim = feature_dim*2
    else:
        fff_dim = feature_dim
    return fff_dim


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
        # pi = pi.clamp(min=-1.0, max=1.0)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        # print(m.weight.shape)
        if len(m.weight.shape) <= 3:
            assert m.weight.size(1) == m.weight.size(2)
            m.weight.data.fill_(0.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
            mid = m.weight.size(2) // 2
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, stride,
        action_type='continuous'
    ):
        super().__init__()

        self.action_type = action_type

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, stride
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        fff_dim = get_dim(encoder_type, self.encoder.feature_dim)

        print("saction.shape:", action_shape)

        self.trunk = nn.Sequential(
            nn.Linear(fff_dim, hidden_dim),
            # nn.Linear(self.encoder.feature_dim // 2 + self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        final_fea, _ = self.encoder(obs, detach=detach_encoder)
        functional.reset_net(self.encoder)

        mu, log_std = self.trunk(final_fea).chunk(2, dim=-1)


        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std

        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if self.action_type == "discrete":
            mu = F.softmax(mu, dim=1)
            mu_dist = Categorical(mu)
            mu = mu_dist.sample().view(-1, 1)
            if compute_pi:
                pi = F.softmax(pi, dim=1)
                pi_dist = Categorical(pi)
                pi = pi_dist.sample().view(-1, 1)

        else:
            pass

        return mu, pi, log_pi, log_std


    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)



class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, stride,
        action_type='continuous'
    ):
        super().__init__()

        self.action_type = action_type
        self.action_shape = action_shape
        fff_dim = get_dim(encoder_type, encoder_feature_dim)

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, stride
        )

        # self.encoder_k = make_encoder(
        #     encoder_type, obs_shape, encoder_feature_dim, num_layers,
        #     num_filters, stride
        # )

        self.W_rgb = nn.Parameter(torch.rand(encoder_feature_dim, encoder_feature_dim))
        self.W_dvs = nn.Parameter(torch.rand(encoder_feature_dim, encoder_feature_dim))
        self.W_com = nn.Parameter(torch.rand(encoder_feature_dim, encoder_feature_dim))

        self.Q1 = QFunction(
            fff_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            fff_dim, action_shape[0], hidden_dim
        )

        # self.fc =  nn.Sequential(
        #     nn.Linear(self.encoder.feature_dim*4, self.encoder.feature_dim*2), nn.ReLU(),
        #     nn.Linear(self.encoder.feature_dim*2, self.encoder.feature_dim), nn.ReLU(),
        #     nn.Linear(self.encoder.feature_dim, self.encoder.feature_dim)
        # )
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        final_fea, _ = self.encoder(obs, detach=detach_encoder)
        functional.reset_net(self.encoder)

        # print("final_fea.shape:", final_fea.shape)
        if self.action_type == 'continuous':
            pass
        elif self.action_type == 'discrete':
            tmp_action = action
            action = torch.zeros(tmp_action.shape[0], self.action_shape[0]).cuda()
            action.scatter_(1, tmp_action.to(torch.int64), 1)
            action = action.to(torch.float32)

        q1 = self.Q1(final_fea, action)
        q2 = self.Q2(final_fea, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        # for k, v in self.outputs.items():
        #     L.log_histogram('train_critic/%s_hist' % k, v, step)
        #
        # for i in range(3):
        #     L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
        #     L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)




