# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
# import info_nce


from utils.soft_update_params import soft_update_params
from utils.preprocess_obs import preprocess_obs

from sac_ae import Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder

plt.switch_backend('agg')


class MMSPRAgent(object):
    """Baseline algorithm with transition model and various decoder types."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        momentum_tau=0.05,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        perception_type='RGB-frame',
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        action_model_update_freq=1,
        transition_reward_model_update_freq=1,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_weight_lambda=0.0,
        transition_model_type='deterministic',
        num_layers=4,
        num_filters=32,
        LOG_FREQ=5000,
    ):
        # self.fig, [self.ax1, self.ax2] = plt.subplots(1, 2)

        self.similarity = nn.CosineSimilarity(dim=1)
        self.momentum_tau = momentum_tau
        self.LOG_FREQ = LOG_FREQ
        self.obs_shape = obs_shape
        self.reconstruction = False
        self.encoder_type = encoder_type
        self.action_model_update_freq = action_model_update_freq
        self.encoder_feature_dim = encoder_feature_dim

        if decoder_type == 'reconstruction':
            decoder_type = 'pixel'
            self.reconstruction = True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.transition_reward_model_update_freq = transition_reward_model_update_freq
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # action embedding
        self.action_emb = nn.Linear(action_shape[0], encoder_feature_dim).to(device)
        # 非对称结构
        self.global_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim * 3, encoder_feature_dim * 3),
            nn.ReLU(),
            nn.Linear(encoder_feature_dim * 3, encoder_feature_dim * 3)
        ).to(device)
        self.global_target_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim * 3, encoder_feature_dim * 3),
            nn.ReLU(),
            nn.Linear(encoder_feature_dim * 3, encoder_feature_dim * 3)
        ).to(device)
        self.global_final_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim * 3, encoder_feature_dim * 3),
            nn.ReLU(),
            nn.Linear(encoder_feature_dim * 3, encoder_feature_dim * 3)
        ).to(device)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.transition_model_rgb = make_transition_model(
            transition_model_type, encoder_feature_dim, encoder_feature_dim, encoder_feature_dim, contain_action=True
        ).to(device)
        self.transition_model_dvs = make_transition_model(
            transition_model_type, encoder_feature_dim, encoder_feature_dim, encoder_feature_dim, contain_action=True
        ).to(device)
        self.transition_model_con = make_transition_model(
            transition_model_type, encoder_feature_dim, encoder_feature_dim, encoder_feature_dim,
            contain_action=True
        ).to(device)
        self.reward_decoder_rgb = nn.Sequential(
            nn.Linear(encoder_feature_dim + encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)
        self.reward_decoder_dvs = nn.Sequential(
            nn.Linear(encoder_feature_dim + encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)
        self.reward_decoder_con = nn.Sequential(
            nn.Linear(encoder_feature_dim + encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        decoder_params = list(self.transition_model_rgb.parameters()) + \
                         list(self.transition_model_dvs.parameters()) + \
                         list(self.transition_model_con.parameters()) + \
                         list(self.reward_decoder_rgb.parameters()) + \
                         list(self.reward_decoder_dvs.parameters()) + \
                         list(self.reward_decoder_con.parameters()) + \
                         list(self.action_emb.parameters()) + \
                         list(self.global_classifier.parameters()) + \
                         list(self.global_final_classifier.parameters()) + \
                         list(self.global_target_classifier.parameters())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)


        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )
        # optimizer for critic encoder for reconstruction loss
        print("self.critic.encoder.parameters():", self.critic.encoder.parameters())
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, 'decoder_rgb'):
            self.decoder_rgb.train(training)
        if hasattr(self, 'decoder_dvs'):
            self.decoder_dvs.train(training)

        # if self.decoder is not None:
        #     self.decoder.train(training)
        # if len(self.decoder) != 0:
        #     for one_decoder in self.decoder:
        #         one_decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, list) and len(obs) == 2:
            rgb_obs = torch.FloatTensor(obs[0]).to(self.device)
            dvs_obs = torch.FloatTensor(obs[1]).to(self.device)
            rgb_obs = rgb_obs.unsqueeze(0)
            dvs_obs = dvs_obs.unsqueeze(0)

            _obs = [rgb_obs, dvs_obs]
        elif isinstance(obs, list) and len(obs) == 3:
            rgb_obs = torch.FloatTensor(obs[0]).to(self.device)
            dvs_obs = torch.FloatTensor(obs[1]).to(self.device)
            depth_obs = torch.FloatTensor(obs[2]).to(self.device)
            rgb_obs = rgb_obs.unsqueeze(0)
            dvs_obs = dvs_obs.unsqueeze(0)
            depth_obs = depth_obs.unsqueeze(0)

            _obs = [rgb_obs, dvs_obs, depth_obs]

        elif isinstance(obs, list) and len(obs) == 4:
            rgb_obs = torch.FloatTensor(obs[0]).to(self.device)
            dvs_obs = torch.FloatTensor(obs[1]).to(self.device)
            depth_obs = torch.FloatTensor(obs[2]).to(self.device)
            dvs_obs2 = torch.FloatTensor(obs[3]).to(self.device)

            rgb_obs = rgb_obs.unsqueeze(0)
            dvs_obs = dvs_obs.unsqueeze(0)
            depth_obs = depth_obs.unsqueeze(0)
            dvs_obs2 = dvs_obs2.unsqueeze(0)

            _obs = [rgb_obs, dvs_obs, depth_obs, dvs_obs2]

        else:
            _obs = torch.FloatTensor(obs).to(self.device)
            _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                _obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            # print("policy_action.nan:", torch.any(torch.isnan(policy_action)))
            # print("target_Q1.nan:", torch.any(torch.isnan(target_Q1)))
            # print("target_Q2.nan:", torch.any(torch.isnan(target_Q2)))

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        # print("current_Q1.nan:", torch.any(torch.isnan(current_Q1)))
        # print("current_Q2.nan:", torch.any(torch.isnan(current_Q2)))

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step, log_freq=self.LOG_FREQ)


    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_consistency(self, obs, action, next_obs, L, step):
        if self.encoder_type == "pixelConNewV4":
            consistency_params = {}  # 用于可视化
            ###############################################################
            _, [rgb_h_query, com_h_query, dvs_h_query] = self.critic.encoder(obs)  # queries: N x z_dim

            with torch.no_grad():
                final_query = self.global_target_classifier(torch.cat([
                    rgb_h_query, com_h_query, dvs_h_query
                ], dim=1))
            rgb_h_query, com_h_query, dvs_h_query = \
                final_query[:, :self.encoder_feature_dim], \
                final_query[:, self.encoder_feature_dim:self.encoder_feature_dim*2], \
                final_query[:, self.encoder_feature_dim*2:]
            # 第一个维度是batch维，要用:
            rgb_h_query = nn.functional.normalize(rgb_h_query, dim=1)
            com_h_query = nn.functional.normalize(com_h_query, dim=1)
            dvs_h_query = nn.functional.normalize(dvs_h_query, dim=1)

            consistency_params['rgb_h_query'] = rgb_h_query.clone().cpu().detach().numpy()
            consistency_params['com_h_query'] = com_h_query.clone().cpu().detach().numpy()
            consistency_params['dvs_h_query'] = dvs_h_query.clone().cpu().detach().numpy()
            ###############################################################

            with torch.no_grad():  # no gradient to keys
                _, [rgb_h_key, com_h_key, dvs_h_key] = self.critic_target.encoder(obs)  # keys: N x z_dim
            final_key = self.global_classifier(torch.cat([
                rgb_h_key, com_h_key, dvs_h_key
            ], dim=1))  # proj
            final_key = self.global_final_classifier(final_key)  # pred
            rgb_h_key, com_h_key, dvs_h_key = \
                final_key[:, :self.encoder_feature_dim], \
                final_key[:, self.encoder_feature_dim:self.encoder_feature_dim * 2], \
                final_key[:, self.encoder_feature_dim * 2:]
            rgb_h_key = nn.functional.normalize(rgb_h_key, dim=1)
            com_h_key = nn.functional.normalize(com_h_key, dim=1)
            dvs_h_key = nn.functional.normalize(dvs_h_key, dim=1)

            consistency_params['rgb_h_key'] = rgb_h_key.clone().cpu().detach().numpy()
            consistency_params['com_h_key'] = com_h_key.clone().cpu().detach().numpy()
            consistency_params['dvs_h_key'] = dvs_h_key.clone().cpu().detach().numpy()

            # temperature = 1.0
            temperature = 0.1

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # V2-1 com和rgb_h/dvs_h拉远，和每个batch都比较
            negative_keys = torch.cat([rgb_h_key, dvs_h_key], dim=0)
            ###############################################################
            positive_logit = torch.sum(com_h_query * com_h_key, dim=1, keepdim=True)
            negative_logits = com_h_query @ (negative_keys.transpose(-2, -1))
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            consistency_params['com_to_else_logits'] = logits.clone().cpu().detach().numpy()
            labels = torch.zeros(len(logits), dtype=torch.long, device=com_h_query.device)
            ###############################################################
            incon_con_diff = F.cross_entropy(logits / temperature, labels, reduction='mean')
            L.log('train_ae/incon_con_diff', incon_con_diff, step)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            incon_con_diff.backward()
            # loss.backward(retain_graph=True)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    
    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixelConNewV4":
            obs, action, _, reward, next_obs, not_done = replay_buffer.sample_dm3dp()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        self.update_consistency(obs, action, next_obs, L, step)


        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.momentum_tau
            )
            if self.encoder_type == "pixelConNewV4":
                soft_update_params(
                    self.global_classifier,
                    self.global_target_classifier,
                    self.momentum_tau
                )


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        # if self.decoder is not None:
        #     torch.save(
        #         self.decoder.state_dict(),
        #         '%s/decoder_%s.pt' % (model_dir, step)
        #     )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        # if self.decoder is not None:
        #     self.decoder.load_state_dict(
        #         torch.load('%s/decoder_%s.pt' % (model_dir, step))
        #     )

