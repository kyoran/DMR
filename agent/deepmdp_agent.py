import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import info_nce
from spikingjelly.activation_based import functional


from utils.soft_update_params import soft_update_params
from utils.preprocess_obs import preprocess_obs

from sac_ae import Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder

plt.switch_backend('agg')


class DeepMDPAgent(object):
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
        action_type='continuous',
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
            num_layers, num_filters, encoder_stride, action_type
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, action_type
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, action_type
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())


        if encoder_type == "pixelConNewV4" or encoder_type == "pixelConNewV4_Repel" or encoder_type == "pixelConNewV4_Rec" \
                or encoder_type == "DMR_SNN" or encoder_type == "DMR_CNN":

            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # 非对称结构
            self.global_classifier = nn.Sequential(
                nn.Linear(encoder_feature_dim*3, encoder_feature_dim*3),
                nn.ReLU(),
                nn.Linear(encoder_feature_dim*3, encoder_feature_dim*3)
            ).to(device)
            self.global_target_classifier = nn.Sequential(
                nn.Linear(encoder_feature_dim*3, encoder_feature_dim*3),
                nn.ReLU(),
                nn.Linear(encoder_feature_dim*3, encoder_feature_dim*3)
            ).to(device)
            self.global_final_classifier = nn.Sequential(
                nn.Linear(encoder_feature_dim*3, encoder_feature_dim*3),
                nn.ReLU(),
                nn.Linear(encoder_feature_dim*3, encoder_feature_dim*3)
            ).to(device)
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            self.transition_model_con = make_transition_model(
                transition_model_type, action_shape[0], encoder_feature_dim, encoder_feature_dim,
                contain_action=True
            ).to(device)
            self.reward_decoder_con = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)

            decoder_params = list(self.transition_model_con.parameters()) + \
                             list(self.reward_decoder_con.parameters()) + \
                             list(self.global_classifier.parameters()) + \
                             list(self.global_final_classifier.parameters()) + \
                             list(self.global_target_classifier.parameters())

            # print("@@@obs_shape:", obs_shape)
            # [(9, 128, 128), (15, 128, 128)]
            self.rec_decoder_rgb = make_decoder(
                "pixel", obs_shape, encoder_feature_dim,
                num_layers, num_filters, obs_shape[0][0]
            ).to(device)
            self.rec_decoder_dvs = make_decoder(
                "pixel", obs_shape, encoder_feature_dim,
                num_layers, num_filters, obs_shape[1][0]
            ).to(device)
            self.rec_decoder_rgb.apply(weight_init)
            self.rec_decoder_dvs.apply(weight_init)
            decoder_params += list(self.rec_decoder_rgb.parameters())
            decoder_params += list(self.rec_decoder_dvs.parameters())


        elif encoder_type == "pixelEFNet" or encoder_type == "pixelFPNNet" or encoder_type == "pixelRENet":

            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape[0], encoder_feature_dim, contain_action=True
            ).to(device)
            self.reward_decoder = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                # nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
            decoder_params = list(self.transition_model.parameters()) + \
                             list(self.reward_decoder.parameters())

            self.decoder = None

        elif encoder_type == "pixelCat" or encoder_type == "pixelCrossFusion" \
                or encoder_type == "pixelFPNNet":
            self.action_emb = nn.Linear(action_shape[0], encoder_feature_dim).to(device)

            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim*2, action_shape[0], encoder_feature_dim*2, contain_action=True
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

        elif encoder_type == "pixelCatSep":
            self.transition_model_rgb = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_dvs = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_cat = make_transition_model(
                transition_model_type, encoder_feature_dim * 2, action_shape, encoder_feature_dim * 2,
                contain_action=True
            ).to(device)
            self.action_model = None
            self.decoder = None
            self.reward_decoder_rgb = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
            self.reward_decoder_dvs = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
            self.reward_decoder_cat = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2 + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)


            decoder_params = list(self.transition_model_rgb.parameters()) + \
                             list(self.transition_model_dvs.parameters()) + \
                             list(self.transition_model_cat.parameters()) + \
                             list(self.reward_decoder_rgb.parameters()) + \
                             list(self.reward_decoder_dvs.parameters()) + \
                             list(self.reward_decoder_cat.parameters())

        else:


            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape[0], encoder_feature_dim, contain_action=True
            ).to(device)

            self.reward_decoder = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
            self.decoder = None

            decoder_params = list(self.transition_model.parameters()) + \
                             list(self.reward_decoder.parameters())

            if perception_type == "RGB-frame":
                outtttt = 9
            elif perception_type == "DVS-voxel-grid":
                outtttt = 15
            else:
                outtttt = 3

            if self.reconstruction:

                self.decoder = make_decoder(
                    decoder_type, obs_shape, encoder_feature_dim,
                    num_layers, num_filters, outtttt
                ).to(device)
                self.decoder.apply(weight_init)
                decoder_params += list(self.decoder.parameters())

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
        # exploit
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                _obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        # explore
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            _, pi, _, _ = self.actor(_obs, compute_pi=True, compute_log_pi=False)
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


    def update_transition_reward_model_pixelDMR(self, obs, action, next_obs, reward, L, step):
        fuse_h, [rgb_h, con_h, dvs_h] = self.critic.encoder(obs)
        functional.reset_net(self.critic.encoder)
        next_fuse_h, [next_rgb_h, next_con_h, next_dvs_h] = self.critic.encoder(next_obs)
        functional.reset_net(self.critic.encoder)

        pred_next_fuse_latent_mu, pred_next_fuse_latent_sigma = self.transition_model_con(torch.cat([con_h, action], dim=1))
        if pred_next_fuse_latent_sigma is None: pred_next_fuse_latent_sigma = torch.ones_like(pred_next_fuse_latent_mu)

        # !!!! 注意下面有加法
        fuse_diff = (pred_next_fuse_latent_mu - next_con_h.detach()) / pred_next_fuse_latent_sigma
        fuse_transition_loss = torch.mean(0.5 * fuse_diff.pow(2) + torch.log(pred_next_fuse_latent_sigma))
        transition_loss = fuse_transition_loss
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_fuse_reward = self.reward_decoder_con(torch.cat([con_h, action], dim=1))
        con_reward_loss = F.mse_loss(pred_next_fuse_reward, reward)
        reward_loss = con_reward_loss
        L.log('train_ae/reward_loss', reward_loss, step)

        total_loss = transition_loss + reward_loss

        return total_loss




    def update_transition_reward_model_pixelHybrid(self, obs, action, next_obs, reward, L, step):

        fusion_h, [rgb_h, dvs_h] = self.critic.encoder(obs)
        next_fusion_h, [next_rgb_h, next_dvs_h] = self.critic.encoder(next_obs)

        # pred_next_rgb_latent_mu, pred_next_rgb_latent_sigma = self.transition_model_rgb(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_latent_mu, pred_next_dvs_latent_sigma = self.transition_model_dvs(torch.cat([dvs_h, action], dim=1))
        # pred_next_cat_latent_mu, pred_next_cat_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        # if pred_next_rgb_latent_sigma is None: pred_next_rgb_latent_sigma = torch.ones_like(pred_next_rgb_latent_mu)
        if pred_next_dvs_latent_sigma is None: pred_next_dvs_latent_sigma = torch.ones_like(pred_next_dvs_latent_mu)
        # if pred_next_cat_latent_sigma is None: pred_next_cat_latent_sigma = torch.ones_like(pred_next_cat_latent_mu)
        # rgb_diff = (pred_next_rgb_latent_mu - next_rgb_h.detach()) / pred_next_rgb_latent_sigma
        dvs_diff = (pred_next_dvs_latent_mu - next_dvs_h.detach()) / pred_next_dvs_latent_sigma
        # cat_diff = (pred_next_cat_latent_mu - next_fusion_h.detach()) / pred_next_cat_latent_sigma
        # rgb_transition_loss = torch.mean(0.5 * rgb_diff.pow(2) + torch.log(pred_next_rgb_latent_sigma))
        dvs_transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_dvs_latent_sigma))
        # cat_transition_loss = torch.mean(0.5 * cat_diff.pow(2) + torch.log(pred_next_cat_latent_sigma))
        transition_loss = dvs_transition_loss# + cat_transition_loss
        L.log('train_ae/transition_loss', transition_loss, step)

        # pred_next_rgb_reward = self.reward_decoder_rgb(torch.cat([rgb_h, action], dim=1))
        # pred_next_dvs_reward = self.reward_decoder_dvs(torch.cat([dvs_h, action], dim=1))
        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        # rgb_reward_loss = F.mse_loss(pred_next_rgb_reward, reward)
        # dvs_reward_loss = F.mse_loss(pred_next_dvs_reward, reward)
        cat_reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        reward_loss = cat_reward_loss
        # reward_loss = T_RGB * rgb_reward_loss + T_DVS * dvs_reward_loss + T_CAT * cat_reward_loss
        L.log('train_ae/reward_loss', reward_loss, step)

        # pred_action = self.action_model(torch.cat([dvs_h, next_dvs_h], dim=1))
        # action_loss = F.mse_loss(pred_action, action)
        # L.log('train_ae/mask_loss', action_loss, step)

        total_loss = transition_loss + reward_loss# + action_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_transition_reward_model_pixelCat(self, obs, action, next_obs, reward, L, step):
        fusion_h, [rgb_h, dvs_h] = self.critic.encoder(obs)
        next_fusion_h, [next_rgb_h, next_dvs_h] = self.critic.encoder(next_obs)

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([fusion_h, action], dim=1))
        if pred_next_latent_sigma is None: pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_fusion_h.detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_reward = self.reward_decoder(torch.cat([fusion_h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)


        total_loss = transition_loss + reward_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def update_transition_reward_model_pixelCatSep(self, obs, action, next_obs, reward, L, step):
        fusion_h, [rgb_h, dvs_h] = self.critic.encoder(obs)
        next_fusion_h, [next_rgb_h, next_dvs_h] = self.critic.encoder(next_obs)

        pred_next_rgb_latent_mu, pred_next_rgb_latent_sigma = self.transition_model_rgb(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_latent_mu, pred_next_dvs_latent_sigma = self.transition_model_dvs(torch.cat([dvs_h, action], dim=1))
        pred_next_cat_latent_mu, pred_next_cat_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        if pred_next_rgb_latent_sigma is None: pred_next_rgb_latent_sigma = torch.ones_like(pred_next_rgb_latent_mu)
        if pred_next_dvs_latent_sigma is None: pred_next_dvs_latent_sigma = torch.ones_like(pred_next_dvs_latent_mu)
        if pred_next_cat_latent_sigma is None: pred_next_cat_latent_sigma = torch.ones_like(pred_next_cat_latent_mu)

        rgb_diff = (pred_next_rgb_latent_mu - next_rgb_h.detach()) / pred_next_rgb_latent_sigma
        dvs_diff = (pred_next_dvs_latent_mu - next_dvs_h.detach()) / pred_next_dvs_latent_sigma
        cat_diff = (pred_next_cat_latent_mu - next_fusion_h.detach()) / pred_next_cat_latent_sigma
        rgb_transition_loss = torch.mean(0.5 * rgb_diff.pow(2) + torch.log(pred_next_rgb_latent_sigma))
        dvs_transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_dvs_latent_sigma))
        cat_transition_loss = torch.mean(0.5 * cat_diff.pow(2) + torch.log(pred_next_cat_latent_sigma))
        transition_loss = rgb_transition_loss + dvs_transition_loss + cat_transition_loss
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_rgb_reward = self.reward_decoder_rgb(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_reward = self.reward_decoder_dvs(torch.cat([dvs_h, action], dim=1))
        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        rgb_reward_loss = F.mse_loss(pred_next_rgb_reward, reward)
        dvs_reward_loss = F.mse_loss(pred_next_dvs_reward, reward)
        cat_reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        reward_loss = rgb_reward_loss + dvs_reward_loss + cat_reward_loss
        L.log('train_ae/reward_loss', reward_loss, step)


        total_loss = transition_loss + reward_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.critic.encoder.parameters(), self.critic_target.encoder.parameters()
        ):
            param_k.data = param_k.data * self.M + param_q.data * (1.0 - self.M)


    def update_consistency(self, obs, action, next_obs, L, step):
        if self.encoder_type == "pixelConNewV4" or self.encoder_type == "DMR_SNN" or self.encoder_type == "DMR_CNN" or self.encoder_type == "pixelConNewV4_Repel":
            consistency_params = {}  # 用于可视化
            ###############################################################
            _, [rgb_h_query, com_h_query, dvs_h_query] = self.critic.encoder(obs)  # queries: N x z_dim
            functional.reset_net(self.critic.encoder)

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
                functional.reset_net(self.critic_target.encoder)

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
            # V2-2 rgb_h和dvs_h拉远，和每个batch都比较
            # [dvs]
            # dvs_positive_logit = torch.sum(dvs_h_query * dvs_h_key, dim=1, keepdim=True)
            # dvs_negative_logits = dvs_h_query @ (rgb_h_key.transpose(-2, -1))
            # logits = torch.cat([dvs_positive_logit, dvs_negative_logits], dim=1)
            # consistency_params['dvs_to_rgb_logits'] = logits.clone().cpu().detach().numpy()
            # labels = torch.zeros(len(logits), dtype=torch.long, device=dvs_h_query.device)
            # dvs_incon_loss = F.cross_entropy(logits / temperature, labels, reduction='mean')
            # [rgb]
            # rgb_positive_logit = torch.sum(rgb_h_query * rgb_h_key, dim=1, keepdim=True)
            # rgb_negative_logits = rgb_h_query @ (dvs_h_key.transpose(-2, -1))
            # logits = torch.cat([rgb_positive_logit, rgb_negative_logits], dim=1)
            # consistency_params['rgb_to_dvs_logits'] = logits.clone().cpu().detach().numpy()
            # labels = torch.zeros(len(logits), dtype=torch.long, device=rgb_h_query.device)
            # rgb_incon_loss = F.cross_entropy(logits / temperature, labels, reduction='mean')
            # inter_incon_diff = dvs_incon_loss + rgb_incon_loss
            # L.log('train_ae/inter_incon_diff', inter_incon_diff, step)
            ###############################################################

            # loss = inter_incon_diff + incon_con_diff
            loss = incon_con_diff
            return loss, consistency_params

    def update_decoder(self, obs, action, target_obs, L, step):

        _, [rgb_h, _, dvs_h] = self.critic.encoder(obs)
        functional.reset_net(self.critic.encoder)

        rec_rgb_obs = self.rec_decoder_rgb(rgb_h)
        rec_dvs_obs = self.rec_decoder_dvs(dvs_h)

        loss = F.mse_loss(target_obs[0], rec_rgb_obs) + \
               F.mse_loss(target_obs[1], rec_dvs_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h, _ = self.critic.encoder(obs)
        functional.reset_net(self.critic.encoder)
        # print("h.nan:", torch.any(torch.isnan(h)))
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 增加action embedding
        # action = self.action_emb(action)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h, _ = self.critic.encoder(next_obs)
        functional.reset_net(self.critic.encoder)
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
        # if step <= 5000:   # airsim
        #     obs, action, _, reward, next_obs, not_done = replay_buffer.sample_dm3dp()

        #     GAMMA = 1.0
        #     trans_loss = self.update_transition_reward_model_pixelDMR(obs, action, next_obs, reward, L, step)
        #     con_loss, consistency_params = self.update_consistency(obs, action, next_obs, L, step)
        #     total_loss = trans_loss + GAMMA * con_loss

        #     self.encoder_optimizer.zero_grad()
        #     self.decoder_optimizer.zero_grad()
        #     total_loss.backward()
        #     # loss.backward(retain_graph=True)
        #     self.encoder_optimizer.step()
        #     self.decoder_optimizer.step()

        #     self.update_decoder(obs, action, next_obs, L, step)
        #     return

        if self.encoder_type == "pixelConNewV4" \
                or self.encoder_type == "pixelConNewV4_Repel" \
                or self.encoder_type == "DMR_SNN" \
                or self.encoder_type == "DMR_CNN" \
                or self.encoder_type == "pixelConNewV4_Rec":
            obs, action, _, reward, next_obs, not_done = replay_buffer.sample_dm3dp()

        else:
            obs, action, _, reward, next_obs, not_done = replay_buffer.sample(
                sep=False, multi=True if isinstance(self.obs_shape, list) and len(self.obs_shape) >= 2 else False)
        
        # if self.encoder_type == "pixelCon":
        #     batch_size = action.shape[0]
        #     row = np.arange(0, batch_size)    
        #     col = np.random.randint(0, 10, batch_size)  
        #
        #     obs = [obs[0][row, col, :, :, :], obs[1][row, col, :, :, :]]
        #     action = action[row, col, :]
        #     next_obs = [next_obs[0][row, col, :, :, :], next_obs[1][row, col, :, :, :]]
        #     reward = reward[row, col, :]
        #     not_done = not_done[row, col, :]


        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.transition_reward_model_update_freq == 0:

            GAMMA = 1.0
            if self.encoder_type == "pixelConNewV4" \
                    or self.encoder_type == "pixelConNewV4_Repel" \
                    or self.encoder_type == "DMR_SNN" \
                    or self.encoder_type == "DMR_CNN" \
                    or self.encoder_type == "pixelConNewV4_Rec":

                trans_loss = self.update_transition_reward_model_pixelDMR(obs, action, next_obs, reward, L, step)

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

            elif self.encoder_type == "pixelCat" or self.encoder_type == "pixelCrossFusion":
                self.update_transition_reward_model_pixelCat(obs, action, next_obs, reward, L, step)

            elif self.encoder_type == "pixelCatSep":
                self.update_transition_reward_model_pixelCatSep(obs, action, next_obs, reward, L, step)
            
            # else:
            #     self.update_transition_reward_model(obs, action, next_obs, reward, L, step)

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
            if self.encoder_type == "pixelConNewV4" \
                    or self.encoder_type == "pixelConNewV4_Repel" \
                    or self.encoder_type == "DMR_SNN" \
                    or self.encoder_type == "DMR_CNN" \
                    or self.encoder_type == "pixelConNewV4_Rec":
                soft_update_params(
                    self.global_classifier,
                    self.global_target_classifier,
                    self.momentum_tau
                )

        # decoder
        if (self.encoder_type == "pixelConNewV4"
            or self.encoder_type == "pixelConNewV4_Rec"
            or self.encoder_type == "DMR_SNN"
            or self.encoder_type == "DMR_CNN") and step % self.decoder_update_freq == 0:
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

