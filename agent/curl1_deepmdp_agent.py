import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
# import info_nce


from utils.soft_update_params import soft_update_params
from utils.preprocess_obs import preprocess_obs

from sac_ae import Actor, Critic, weight_init, LOG_FREQ, CURLHead
from transition_model import make_transition_model
from decoder import make_decoder

import augmentations

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
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
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

        if encoder_type == "pixelCon":

            # v1-v4：有negative的时候
            # self.info_nce = info_nce.InfoNCE(negative_mode='unpaired')
            # v5, v6：没有negative
            self.info_nce = info_nce.InfoNCE()

            ###########################

            # create the queue
            self.K = 1000
            self.T = 0.07
            self.M = 0.999
            # self.register_buffer("queue_rgb", torch.randn(encoder_feature_dim, self.K))
            # self.register_buffer("queue_dvs", torch.randn(encoder_feature_dim, self.K))
            # self.queue_rgb = nn.functional.normalize(self.queue_rgb, dim=0)
            # self.queue_dvs = nn.functional.normalize(self.queue_dvs, dim=0)
            # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            ##########################


            self.transition_model_rgb = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_dvs = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_con = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
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
            self.reward_decoder_con = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)

            decoder_params = list(self.transition_model_rgb.parameters()) + \
                             list(self.transition_model_dvs.parameters()) + \
                             list(self.transition_model_con.parameters()) + \
                             list(self.reward_decoder_rgb.parameters()) + \
                             list(self.reward_decoder_dvs.parameters()) + \
                             list(self.reward_decoder_con.parameters())

            self.decoder = None




        elif encoder_type == "pixelCat":

            self.transition_model_rgb2 = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_dvs2 = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)

            self.reward_decoder_con = nn.Sequential(
                nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)

            # decoder_params = list(self.transition_model_rgb.parameters()) + \
            #                  list(self.transition_model_dvs.parameters()) + \
            #                  list(self.transition_model_con.parameters()) + \
            #                  list(self.reward_decoder_rgb.parameters()) + \
            #                  list(self.reward_decoder_dvs.parameters()) + \
            #                  list(self.reward_decoder_con.parameters())
            decoder_params = list(self.transition_model_rgb2.parameters()) + \
                             list(self.transition_model_dvs2.parameters()) + \
                             list(self.reward_decoder_con.parameters())
            self.decoder = None

            if self.reconstruction:

                self.decoder_rgb = make_decoder(
                    decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 15
                ).to(device)
                self.decoder_dvs = make_decoder(
                    # decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 1
                    decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 9  # 5*3
                ).to(device)
                self.decoder_rgb.apply(weight_init)
                self.decoder_dvs.apply(weight_init)

                decoder_params += list(self.decoder_rgb.parameters())
                decoder_params += list(self.decoder_dvs.parameters())


        else:

            self.action_model = None
            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim
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
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder_q)

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
        print("self.critic.encoder_q.parameters():", self.critic.encoder_q.parameters())
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder_q.parameters(), lr=encoder_lr
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




        self.curl_head = CURLHead(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)
        self.curl_head.encoder.copy_conv_weights_from(self.critic.encoder_q)

        self.curl_optimizer = torch.optim.Adam(
            self.curl_head.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )


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


    def update_transition_reward_model_pixelCat(self, obs, action, next_obs, reward, L, step):
        con_h, [rgb_h, dvs_h] = self.critic.encoder_q(obs)
        next_con_h, [next_rgb_h, next_dvs_h] = self.critic.encoder_q(next_obs)



        # pred_next_rgb_latent_mu, pred_next_rgb_latent_sigma = self.transition_model_rgb(torch.cat([dvs_h, action], dim=1))
        # pred_next_dvs_latent_mu, pred_next_dvs_latent_sigma = self.transition_model_dvs(torch.cat([rgb_h, action], dim=1))
        # # pred_next_con_latent_mu, pred_next_con_latent_sigma = self.transition_model_con(torch.cat([con_h, action], dim=1))
        # if pred_next_rgb_latent_sigma is None: pred_next_rgb_latent_sigma = torch.ones_like(pred_next_rgb_latent_mu)
        # if pred_next_dvs_latent_sigma is None: pred_next_dvs_latent_sigma = torch.ones_like(pred_next_dvs_latent_mu)
        # # if pred_next_con_latent_sigma is None: pred_next_con_latent_sigma = torch.ones_like(pred_next_con_latent_mu)


        # rgb_diff = (pred_next_rgb_latent_mu - next_rgb_h.detach()) / pred_next_rgb_latent_sigma
        # dvs_diff = (pred_next_dvs_latent_mu - next_dvs_h.detach()) / pred_next_dvs_latent_sigma
        # # con_diff = (pred_next_con_latent_mu - next_con_h.detach()) / pred_next_con_latent_sigma
        # rgb_transition_loss = torch.mean(0.5 * rgb_diff.pow(2) + torch.log(pred_next_rgb_latent_sigma))
        # dvs_transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_dvs_latent_sigma))
        # # con_transition_loss = torch.mean(0.5 * con_diff.pow(2) + torch.log(pred_next_con_latent_sigma))
        # transition_loss = rgb_transition_loss + dvs_transition_loss# + con_transition_loss
        # L.log('train_ae/transition_loss', transition_loss, step)



        pred_next_rgb_latent_mu, pred_next_rgb_latent_sigma = self.transition_model_rgb2(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_latent_mu, pred_next_dvs_latent_sigma = self.transition_model_dvs2(torch.cat([dvs_h, action], dim=1))
        # pred_next_con_latent_mu, pred_next_con_latent_sigma = self.transition_model_con(torch.cat([con_h, action], dim=1))
        if pred_next_rgb_latent_sigma is None: pred_next_rgb_latent_sigma = torch.ones_like(pred_next_rgb_latent_mu)
        if pred_next_dvs_latent_sigma is None: pred_next_dvs_latent_sigma = torch.ones_like(pred_next_dvs_latent_mu)
        # if pred_next_con_latent_sigma is None: pred_next_con_latent_sigma = torch.ones_like(pred_next_con_latent_mu)


        rgb_diff = (pred_next_rgb_latent_mu - next_rgb_h.detach()) / pred_next_rgb_latent_sigma
        dvs_diff = (pred_next_dvs_latent_mu - next_dvs_h.detach()) / pred_next_dvs_latent_sigma
        # con_diff = (pred_next_con_latent_mu - next_con_h.detach()) / pred_next_con_latent_sigma
        rgb_transition_loss = torch.mean(0.5 * rgb_diff.pow(2) + torch.log(pred_next_rgb_latent_sigma))
        dvs_transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_dvs_latent_sigma))
        # con_transition_loss = torch.mean(0.5 * con_diff.pow(2) + torch.log(pred_next_con_latent_sigma))
        transition_loss2 = rgb_transition_loss + dvs_transition_loss# + con_transition_loss
        L.log('train_ae/transition_loss2', transition_loss2, step)


        # pred_next_rgb_reward = self.reward_decoder_rgb(torch.cat([rgb_h, action], dim=1))
        # pred_next_dvs_reward = self.reward_decoder_dvs(torch.cat([dvs_h, action], dim=1))
        pred_next_con_reward = self.reward_decoder_con(torch.cat([con_h, action], dim=1))
        # rgb_reward_loss = F.mse_loss(pred_next_rgb_reward, reward)
        # dvs_reward_loss = F.mse_loss(pred_next_dvs_reward, reward)
        con_reward_loss = F.mse_loss(pred_next_con_reward, reward)
        reward_loss =   con_reward_loss
        L.log('train_ae/reward_loss', reward_loss, step)


        total_loss =  reward_loss +transition_loss2

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_decoder(self, obs, action, target_obs, L, step):  #  uses transition model

        if self.encoder_type == "pixelCat":

            f, [rgb_h, dvs_h] = self.critic.encoder_q(obs)
            rec_rgb = self.decoder_rgb(rgb_h)
            rec_dvs = self.decoder_dvs(dvs_h)
            loss = F.mse_loss(obs[0], rec_rgb) + F.mse_loss(obs[1], rec_dvs)
            L.log('train_ae/ae_loss', loss, step)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            self.decoder_rgb.log(L, step, log_freq=LOG_FREQ)
            self.decoder_dvs.log(L, step, log_freq=LOG_FREQ)

            return


        else:

            assert target_obs.dim() == 4
            target_obs = target_obs[:, :3, :, :]

            h = self.critic.encoder_q(obs)

            if not self.reconstruction:
                next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
                if target_obs.dim() == 4:
                    # preprocess images to be in [-0.5, 0.5] range
                    target_obs = preprocess_obs(target_obs)
                rec_obs = self.decoder(next_h)
                loss = F.mse_loss(target_obs, rec_obs)
            else:
                rec_obs = self.decoder(h)
                loss = F.mse_loss(obs, rec_obs)


        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h, _ = self.critic.encoder_q(obs)
        # print("h.nan:", torch.any(torch.isnan(h)))

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h, _ = self.critic.encoder_q(next_obs)
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


    def update_curl(self, x, x_pos, L=None, step=None):
        # assert x.size(-1) == 84 and x_pos.size(-1) == 84

        z_a = self.curl_head.encoder(x)
        with torch.no_grad():
            z_pos = self.critic_target.encoder_q(x_pos)
            

        logits = self.curl_head.compute_logits(z_a[0], z_pos[0])
        labels = torch.arange(logits.shape[0]).long().cuda()
        curl_loss = F.cross_entropy(logits, labels)
        
        self.curl_optimizer.zero_grad()
        curl_loss.backward()
        self.curl_optimizer.step()
        if L is not None:
            L.log('train/aux_loss', curl_loss, step)


    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample(
            multi=True if isinstance(self.obs_shape, list) and len(self.obs_shape) >= 2 else False)
        

        #pos = augmentations.random_crop_2(obs.clone())
        #obs = augmentations.random_crop_2(obs)
        #next_obs = augmentations.random_crop_2(next_obs)
        
        
        pos = augmentations.random_crop_2(obs[0].clone())
        obs[0] = augmentations.random_crop_2(obs[0])
        next_obs[0] = augmentations.random_crop_2(next_obs[0])
        
        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        # if step % self.transition_reward_model_update_freq == 0:


        #     if self.encoder_type == "pixelCat":
        #         self.update_transition_reward_model_pixelCat(obs, action, next_obs, reward, L, step)

        #     # 默认单模态
        #     else:
        #         self.update_transition_reward_model(obs, action, next_obs, reward, L, step)

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
                self.critic.encoder_q, self.critic_target.encoder_q,
                self.encoder_tau
            )
        #self.update_curl(obs, pos, L, step)
        self.update_curl(obs[0], pos, L, step)



        # if self.decoder is not None and step % self.decoder_update_freq == 0:
        #     self.update_decoder(obs, action, next_obs, L, step)

        # # self.update_semantic(obs, action, next_obs, L, step)


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

