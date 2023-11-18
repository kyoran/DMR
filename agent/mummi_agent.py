# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import info_nce


from utils.soft_update_params import soft_update_params
from utils.preprocess_obs import preprocess_obs

from sac_ae import Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder

plt.switch_backend('agg')


class MuMMIAgent(object):
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
        embed_viz_dir=None,
    ):
        # self.fig, [self.ax1, self.ax2] = plt.subplots(1, 2)
        self.embed_viz_dir = embed_viz_dir
        self.similarity = nn.CosineSimilarity(dim=1)

        self.LOG_FREQ = LOG_FREQ
        self.obs_shape = obs_shape
        self.reconstruction = False
        self.encoder_type = encoder_type
        self.action_model_update_freq = action_model_update_freq

        if decoder_type == 'reconstruction':
            decoder_type = 'pixel'
            self.reconstruction = True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.momentum_tau = momentum_tau
        self.transition_reward_model_update_freq = transition_reward_model_update_freq
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type
        self.encoder_feature_dim = encoder_feature_dim

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



        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim*2, action_shape, encoder_feature_dim*2, contain_action=True
        ).to(device)
        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
            # nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)
        decoder_params = list(self.transition_model.parameters()) + \
                         list(self.reward_decoder.parameters())

        self.decoder = None

        if self.reconstruction:

            self.decoder_rgb = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 9
            ).to(device)
            self.decoder_dvs = make_decoder(
                # decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 1
                decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 15  # 5*3
            ).to(device)
            self.decoder_rgb.apply(weight_init)
            self.decoder_dvs.apply(weight_init)

            decoder_params += list(self.decoder_rgb.parameters())
            decoder_params += list(self.decoder_dvs.parameters())


        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # self.decoder = None
        # if decoder_type == 'pixel':
        #     # create decoder
        #     if encoder_type == "pixelHybridActionMaskV5":
        #         decoder_type = "HybridActionMaskV5Decoder"
        #
        #     self.decoder = make_decoder(
        #         decoder_type, obs_shape,
        #         encoder_feature_dim if encoder_type != "pixelMultiHeadHybridMask" else encoder_feature_dim * 2,
        #         num_layers,
        #         num_filters
        #     ).to(device)
        #     self.decoder.apply(weight_init)
        #     decoder_params += list(self.decoder.parameters())

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



    def update_decoder(self, obs, action, target_obs, L, step):

        _, [rgb_h, com_h, dvs_h] = self.critic.encoder(obs)

        rec_rgb_obs = self.rec_decoder_rgb(rgb_h)
        rec_dvs_obs = self.rec_decoder_dvs(dvs_h)

        # print("@@@rec_rgb_obs:", rec_rgb_obs.shape)
        # print("@@@rec_dvs_obs:", rec_dvs_obs.shape)
        # print("@@@target_obs[0]:", target_obs[0].shape)
        # print("@@@target_obs[1]:", target_obs[1].shape)

        loss = F.mse_loss(target_obs[0], rec_rgb_obs) + \
               F.mse_loss(target_obs[1], rec_dvs_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        # self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h, _ = self.critic.encoder(obs)
        # print("h.nan:", torch.any(torch.isnan(h)))
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 增加action embedding
        # action = self.action_emb(action)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h, _ = self.critic.encoder(next_obs)
        # print("next_h.nan:", torch.any(torch.isnan(next_h)))
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        # print("pred_next_latent_mu:", pred_next_latent_mu)
        # print("pred_next_latent_sigma:", pred_next_latent_sigma)
        # print("next_h:", next_h)
        # print("diff:", diff)
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        # print("diff.pow(2):", diff.pow(2))
        # print("transition_loss:", transition_loss)
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_reward = self.reward_decoder(torch.cat([h, action], dim=1))
        # print("pred_next_reward:", pred_next_reward)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        # print("reward_loss:", reward_loss)

        L.log('train_ae/reward_loss', reward_loss, step)

        total_loss = transition_loss + reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def update(self, replay_buffer, L, step):
        # obs, action, _, reward, next_obs, not_done = replay_buffer.sample(
        #     sep=True if self.encoder_type == "pixelCon" else False,
        #     multi=True if isinstance(self.obs_shape, list) and
        #                   len(self.obs_shape) >= 2 else False)
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample(
            sep=False, multi=True if isinstance(self.obs_shape, list) and len(self.obs_shape) >= 2 else False)


        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.transition_reward_model_update_freq == 0:

            GAMMA = 1.0
            if self.encoder_type == "pixelConNewV4" or self.encoder_type == "pixelConNewV4_Repel":

                trans_loss = self.update_transition_reward_model_pixelCon(obs, action, next_obs, reward, L, step)

                if GAMMA != 0:
                    con_loss, consistency_params = self.update_consistency(obs, action, next_obs, L, step)
                    if self.embed_viz_dir is not None and step % 10000 == 0:
                        torch.save(
                            consistency_params,
                            os.path.join(self.embed_viz_dir, 'consistency_params_{}.pt'.format(step)))
                    # print("trans_loss:", trans_loss.item(), "con_loss:", con_loss.item())
                    total_loss = trans_loss + GAMMA * con_loss
                else:
                    total_loss = trans_loss

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                total_loss.backward()
                # loss.backward(retain_graph=True)
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()


            elif self.encoder_type == "pixelHybrid":
                self.update_transition_reward_model_pixelHybrid(obs, action, next_obs, reward, L, step)

            elif self.encoder_type == "pixelCat" or self.encoder_type == "pixelCrossFusion":
                self.update_transition_reward_model_pixelCat(obs, action, next_obs, reward, L, step)

            elif self.encoder_type == "pixelCatSep":
                self.update_transition_reward_model_pixelCatSep(obs, action, next_obs, reward, L, step)
            # 默认单模态
            else:
                self.update_transition_reward_model(obs, action, next_obs, reward, L, step)

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
        # decoder
        if self.encoder_type == "pixelConNewV4" and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, action, next_obs, L, step)


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

        if self.encoder_type == "pixelConNewV4":
            torch.save(
                self.rec_decoder_rgb.state_dict(),
                '%s/rec_decoder_rgb_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.rec_decoder_dvs.state_dict(),
                '%s/rec_decoder_dvs%s.pt' % (model_dir, step)
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

