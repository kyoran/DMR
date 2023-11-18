"""
Mask-based Latent Reconstruction for Reinforcement Learning
NIPS 2022
"""



import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import info_nce


from kornia.augmentation import (CenterCrop, RandomAffine, RandomCrop,
                                 RandomResizedCrop)
from kornia.filters import GaussianBlur2d


from utils.soft_update_params import soft_update_params
from utils.preprocess_obs import preprocess_obs

from sac_ae import Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder
import torchvision.transforms._transforms_video as v_transform


from utils.PositionalEmbedding import PositionalEmbedding
from utils.InverseSquareRootSchedule import InverseSquareRootSchedule
from utils.AnneallingSchedule import AnneallingSchedule
from utils.CubeMaskGenerator import CubeMaskGenerator
from utils.vit_modules import Block, trunc_normal_

plt.switch_backend('agg')

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images

class MTM(nn.Module):
    def __init__(self, critic, augmentation, aug_prob, encoder_feature_dim,
                 latent_dim, num_attn_layers, num_heads, device, mask_ratio, jumps, action_shape,
                 patch_size, block_size):
        super().__init__()
        self.aug_prob = aug_prob
        self.device = device
        self.jumps = jumps

        img_size = 128
        input_size = img_size // patch_size
        self.masker = CubeMaskGenerator(
            input_size=input_size, image_size=img_size, clip_size=self.jumps + 1, \
            block_size=block_size, mask_ratio=mask_ratio)  # 1 for mask, num_grid=input_size

        self.position = PositionalEmbedding(encoder_feature_dim)
        # self.position = nn.Parameter(torch.zeros(1, jumps+1, encoder_feature_dim))

        self.state_mask_token = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))
        self.action_mask_token = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))

        # self.state_flag = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))
        # self.action_flag = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))

        self.encoder = critic.encoder
        self.target_encoder = copy.deepcopy(critic.encoder)
        self.global_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))

        self.transformer = nn.ModuleList([
            Block(encoder_feature_dim, num_heads, mlp_ratio=2.,
                  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                  drop_path=0., init_values=0., act_layer=nn.GELU,
                  norm_layer=nn.LayerNorm, attn_head_dim=None)
            for _ in range(num_attn_layers)])
        self.action_emb = nn.Linear(action_shape[0], encoder_feature_dim)
        self.action_predictor = nn.Sequential(
            nn.Linear(encoder_feature_dim, encoder_feature_dim * 2), nn.ReLU(),
            nn.Linear(encoder_feature_dim * 2, action_shape[0])
        )

        ''' Data augmentation '''
        self.intensity = Intensity(scale=0.05)
        self.transforms = []
        self.eval_transforms = []
        self.uses_augmentation = True
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1),
                                              (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4),
                                               RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.apply(self._init_weights)
        # trunc_normal_(self.position, std=.02)
        trunc_normal_(self.state_mask_token, std=.02)
        trunc_normal_(self.action_mask_token, std=.02)
        # trunc_normal_(self.state_flag, std=.02)
        # trunc_normal_(self.action_flag, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image,
                                              transform,
                                              eval_transform,
                                              p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        # images = images.float(
        # ) / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None, flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def spr_loss(self, latents, target_latents, observation, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = self.global_classifier(latents)  # proj
                global_latents = self.global_final_classifier(
                    global_latents)  # pred
        else:
            global_latents = self.global_classifier(latents)  # proj
            global_latents = self.global_final_classifier(
                global_latents)  # pred

        with torch.no_grad():
            global_targets = self.global_target_classifier(target_latents)
        # targets = global_targets.view(-1, observation.shape[1], self.jumps + 1,
        #                               global_targets.shape[-1]).transpose(
        #                                   1, 2)
        # latents = global_latents.view(-1, observation.shape[1], self.jumps + 1,
        #                               global_latents.shape[-1]).transpose(
        #                                   1, 2)
        # loss = self.norm_mse_loss(latents, targets, mean=False)
        loss = self.norm_mse_loss(global_latents, global_targets, mean=False).mean()
        # split to [jumps, bs]
        # return loss.view(-1, observation.shape[1])
        return loss

    def norm_mse_loss(self, f_x1s, f_x2s, mean=True):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1,
                           eps=1e-3)  # (bs*(1+jumps), 512)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        loss = loss.mean(0) if mean else loss
        return loss


class MLRAgent(object):
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            augmentation=["intensity"],
            transition_model_type='deterministic',
            transition_model_layer_width=512,
            jumps=6,
            latent_dim=100,
            time_offset=0,
            momentum_tau=0.05,
            aug_prob=1.0,
            auxiliary_task_lr=1e-3,
            action_aug_type='random',
            num_aug_actions=10,
            loss_space='y',
            bp_mode='gt',
            cycle_steps=6,
            cycle_mode='fp+cycle',
            fp_loss_weight=6.0,
            bp_loss_weight=1.0,
            rc_loss_weight=0.0,
            vc_loss_weight=1.0,
            reward_loss_weight=0.0,
            # from curl
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.1,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_stride=2,
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            cpc_update_freq=1,
            log_interval=100,
            detach_encoder=False,
            curl_latent_dim=128,
            sigma=0.05,
            mask_ratio=0.5,
            patch_size=16,
            block_size=3,
            num_attn_layers=2,
            LOG_FREQ = 5000,
    ):

        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.encoder_feature_dim = encoder_feature_dim
        self.LOG_FREQ = LOG_FREQ

        self.jumps = jumps
        self.momentum_tau = momentum_tau


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

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)


        # optimizer for critic encoder for reconstruction loss
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

        ''' MTM '''
        num_heads = 1
        self.MTM = MTM(self.critic, augmentation, aug_prob, encoder_feature_dim,
                       latent_dim, num_attn_layers, num_heads, device, mask_ratio, jumps,
                       action_shape, patch_size, block_size).to(device)
        self.mtm_optimizer = torch.optim.Adam(self.MTM.parameters(), lr=0.5 * auxiliary_task_lr)
        warmup = True
        adam_warmup_step = 6e3
        encoder_annealling = False
        if warmup:
            lrscheduler = InverseSquareRootSchedule(adam_warmup_step)
            lrscheduler_lambda = lambda x: lrscheduler.step(x)
            self.mtm_lrscheduler = torch.optim.lr_scheduler.LambdaLR(self.mtm_optimizer, lrscheduler_lambda)
            if encoder_annealling:
                lrscheduler2 = AnneallingSchedule(adam_warmup_step)
                lrscheduler_lambda2 = lambda x: lrscheduler2.step(x)
                self.encoder_lrscheduler = torch.optim.lr_scheduler.LambdaLR(self.encoder_optimizer,
                                                                             lrscheduler_lambda2)
            else:
                self.encoder_lrscheduler = None
        else:
            self.mtm_lrscheduler = None
        self.video_crop = v_transform.RandomCropVideo(self.image_size)
        ''' MTM '''

        self.train()
        self.critic_target.train()
        self.MTM.train()

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


    def update_mtm(self, mtm_kwargs, L, step):
        observation = mtm_kwargs["observation"]  # [1+self.jumps, B, 9, 1, 100, 100]
        # print("@@@@@@@ observation:", observation.shape)

        action = mtm_kwargs["action"]  # [1+self.jumps, B, dim_A]
        # print("@@@@@@@ action:", action.shape)

        reward = mtm_kwargs["reward"]  # [1+self.jumps, 1]
        # print("@@@@@@@ reward:", reward.shape)

        T, B, C = observation.size()[:3]
        Z = self.encoder_feature_dim

        position = self.MTM.position(T).transpose(0, 1).to(self.device)  # (1, T, Z) -> (T, 1, Z)
        # print("@@@@@@ position:", position.shape)
        expand_pos_emb = position.expand(T, B, -1)  # (T, B, Z)

        mask = self.MTM.masker()  # (T, 1, 84, 84)
        # print("@@@@@@@ mask1:", mask.shape)
        mask = mask[:, None].expand(mask.size(0), B, *mask.size()[1:]).flatten(0, 1)  # (T*B, ...)
        # print("@@@@@@@ mask2:", mask.shape)

        x = observation.squeeze(-3).flatten(0, 1)
        # print("@@@@@@@ x1:", x.shape)

        x = x * (1 - mask.float().to(self.device))
        x = self.MTM.transform(x, augment=True)

        # print("@@@@@@@ x2:", x.shape)

        x, _ = self.MTM.encoder(x)
        x = x.view(T, B, Z)

        a_vis = action
        a_vis_size = a_vis.size(0)
        a_vis = self.MTM.action_emb(a_vis.flatten(0, 1)).view(a_vis_size, B, Z)

        x_full = torch.zeros(2 * T, B, Z).to(self.device)
        x_full[::2] = x + expand_pos_emb
        x_full[1::2] = a_vis + expand_pos_emb

        x_full = x_full.transpose(0, 1)
        for i in range(len(self.MTM.transformer)):
            x_full = self.MTM.transformer[i](x_full)
        # x_full = self.trans_ln(x_full)
        x_full = x_full.transpose(0, 1)

        pred_masked_s = x_full[::2].flatten(0, 1)  # (M*B, Z)

        target_obs = observation.squeeze(-3).flatten(0, 1)
        target_obs = self.MTM.transform(target_obs, augment=True)
        with torch.no_grad():
            target_masked_s, _ = self.MTM.target_encoder(target_obs)
        state_loss = self.MTM.spr_loss(pred_masked_s, target_masked_s, observation)

        loss = state_loss

        self.mtm_optimizer.zero_grad()
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.MTM.parameters(), 10)
        self.mtm_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/mtm_loss', loss, step)

        if self.mtm_lrscheduler is not None:
            self.mtm_lrscheduler.step()
            L.log('train/mtm_lr', self.mtm_optimizer.param_groups[0]['lr'], step)
            # if self.encoder_lrscheduler is not None:
            #     self.encoder_lrscheduler.step()
            #     L.log('train/ctmr_encoder_lr', self.encoder_optimizer.param_groups[0]['lr'], step)

    def update(self, replay_buffer, L, step):
        # obs, action, _, reward, next_obs, not_done = replay_buffer.sample(
        #     sep=True if self.encoder_type == "pixelCon" else False,
        #     multi=True if isinstance(self.obs_shape, list) and
        #                   len(self.obs_shape) >= 2 else False)

        elements = replay_buffer.sample_spr()
        obs, action, reward, next_obs, not_done, mtm_kwargs = elements

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_mtm(mtm_kwargs, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1,
                               self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2,
                               self.critic_tau)
            soft_update_params(self.critic.encoder,
                               self.critic_target.encoder,
                               self.encoder_tau)
            soft_update_params(self.MTM.encoder,
                               self.MTM.target_encoder,
                               self.momentum_tau)
            soft_update_params(self.MTM.global_classifier,
                               self.MTM.global_target_classifier,
                               self.momentum_tau)

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

