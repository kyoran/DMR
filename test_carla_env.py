# !/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2
import gym
import sys
import math
import time
import json
import glob
import queue
import carla
import shutil
import random
import imageio
import traceback
import distutils
import argparse
import numpy as np
from dotmap import DotMap
from termcolor import colored

from collections import deque
from collections import defaultdict

import torch
import torchvision
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

# from env.CARLA_0_9_13.CarlaEnv_mix import CarlaEnv
from env.CARLA_0_9_13_pithy.CarlaEnv_mix import CarlaEnv
# from env.CARLA_0_9_13_pithy.CarlaEnv_evolve import CarlaEnv

from agent.deepmdp_agent import DeepMDPAgent

from utils.Logger import Logger
from utils.dotdict import dotdict
from utils.make_dir import make_dir
from utils.FrameStack import FrameStack
from utils.ReplayBuffer import ReplayBuffer
from utils.VideoRecorder import VideoRecorder
from utils.seed_everywhere import seed_everywhere

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--suit', default='carla', choices=['carla', 'airsim'])
    ###########################---scenario---###############################
    # parser.add_argument('--domain_name', default='normal')

    # parser.add_argument('--domain_name', default='highbeam')
    # parser.add_argument('--domain_name', default='tunnel')

    # parser.add_argument('--domain_name', default='highspeed')
    # parser.add_argument('--domain_name', default='superspeed')
    #
    parser.add_argument('--domain_name', default='jaywalk')
    # parser.add_argument('--domain_name', default='crash')

    # parser.add_argument('--domain_name', default='narrow')
    ###########################---weather---###############################
    # parser.add_argument('--selected_weather', default='hard_high_light')
    # parser.add_argument('--selected_weather', default='normal')
    # parser.add_argument('--selected_weather', default='cloudy')
    parser.add_argument('--selected_weather', default='midnight')
    # parser.add_argument('--selected_weather', default='hard_rain')
    # parser.add_argument('--selected_weather', default='dense_fog')
    #######################################################################


    parser.add_argument('--agent', default='deepmdp', type=str, choices=['baseline', 'bisim', 'deepmdp'])
    # gpu
    parser.add_argument('--device', default="gpu", type=str)
    parser.add_argument('--gpu_id', default="1", type=str)
    # CARLA 0.9.13
    parser.add_argument('--max_fps', default=200, type=int)
    parser.add_argument('--min_fps', default=20, type=int)
    parser.add_argument('--carla_rpc_port', default=7632, type=int)
    parser.add_argument('--carla_tm_port', default=5631, type=int)
    parser.add_argument('--carla_timeout', default=120, type=int)
    # environment
    parser.add_argument('--work_dir', default='./test_data/test_scenario/', type=str)
    parser.add_argument('--perception_type', default='RGB-Frame+DVS-Voxel-Grid',
                        # choices=['RGB-frame',
                        #          'DVS-frame',
                        #          'DVS-stream',
                        #          'DVS-voxel-grid',
                        #          'E2VID-frame',
                        #          'RGB-Frame+DVS-Frame',
                        #          'RGB-Frame+DVS-Voxel-Grid',
                        #          'MMMI',
                        #          ]
                        )

    parser.add_argument('--SEED', default=333, type=int)
    parser.add_argument('--do_carla_metrics', default=True)
    parser.add_argument('--LOG_FREQ', default=10000, type=int)
    parser.add_argument('--EVAL_FREQ', default=50, type=int)
    parser.add_argument('--SAVE_MODEL_FREQ', default=20, type=int)
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    parser.add_argument('--min_stuck_steps', default=100, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--fov', default=60, type=int)
    parser.add_argument('--rl_image_size', default=256, type=int)
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--frame_skip', default=1, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=[
        'color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    # parser.add_argument('--DVS_norm', default=0, type=int)
    parser.add_argument('--action_model_update_freq', default=1, type=int)
    parser.add_argument('--transition_reward_model_update_freq', default=1, type=int)
    parser.add_argument('--encoder_type', default='pixelHybridEasy', type=str,
                        choices=['pixel', 'pixelCarla096', 'pixelCarla098',
                                 'identity', 'pixelres',
                                 'pixelHybridEasy', 'pixelHybridSet',
                                 'pixelBaselineCat', 'pixelBaselineMaskCat',
                                 'pixelFusedMask', 'pixelFusedMaskAll',
                                 'pixelDVSMask', 'pixelDVSMaskAll',
                                 'pixelDVSMaskV2',
                                 ])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str,
                        choices=['pixel', 'identity', 'contrastive',
                                 'reward', 'inverse', 'reconstruction',
                                 'pixelHybridEasy'
                                 ])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--TPV', default=False, action='store_true')    # third-person view
    parser.add_argument('--BEV', default=False, action='store_true')    # bird-eye view
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--do_metrics', default=False, action='store_true')
    parser.add_argument('--is_spectator', default=True, action='store_true')
    parser.add_argument('--DENOISE', default=False, action='store_true')
    args = parser.parse_args()

    # args.work_dir = args.work_dir + f"{args.perception_type}-{args.agent}-{args.selected_scenario}-{args.selected_weather}"

    return args


def make_env(args, device):
    if args.suit == 'carla':
        with open('./env/CARLA_0_9_13_pithy/weather.json', 'r', encoding='utf8') as fff:
            weather_params = json.load(fff)
        with open('./env/CARLA_0_9_13_pithy/scenario.json', 'r', encoding='utf8') as fff:
            scenario_params = json.load(fff)
        # with open('./tools/rpg_e2vid/dvs_rec_args.json', 'r', encoding='utf8') as fff:
        #     dvs_rec_args = json.load(fff)
        #     dvs_rec_args = dotdict(dvs_rec_args)

        env = CarlaEnv(
            weather_params=weather_params,
            scenario_params=scenario_params,
            selected_scenario=args.domain_name,
            selected_weather=args.selected_weather,
            # selected_speed=args.selected_speed,
            carla_rpc_port=args.carla_rpc_port,
            carla_tm_port=args.carla_tm_port,
            carla_timeout=args.carla_timeout,
            perception_type=args.perception_type,
            num_cameras=args.num_cameras,
            rl_image_size=args.rl_image_size,
            fov=args.fov,
            device=device,
            max_fps=args.max_fps,
            min_fps=args.min_fps,
            min_stuck_steps=args.min_stuck_steps,
            max_episode_steps=args.max_episode_steps,
            frame_skip=args.frame_skip,
            ego_auto_pilot=False,
            # ego_auto_pilot=True,
            DENOISE=args.DENOISE,
            TPV=args.TPV,
            BEV=args.BEV,
            is_spectator=args.is_spectator
        )


        if args.encoder_type.startswith('pixel'):
            env = FrameStack(env, k=args.frame_stack, DENOISE=args.DENOISE, type=args.perception_type)
            # eval_env = utils.FrameStack(eval_env, k=args.frame_stack, type=args.perception_type)

        eval_env = env


    elif args.suit == 'airsim':
        pass

    print("env.observation_space:", env.observation_space)
    print("env.action_space:", env.action_space)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env, eval_env

def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'deepmdp':
        agent = DeepMDPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            action_model_update_freq=args.action_model_update_freq,
            transition_reward_model_update_freq=args.transition_reward_model_update_freq,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )

    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder)
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()

    seed_everywhere(args.SEED)
    device = torch.device(f'cuda' if torch.cuda.is_available() and args.device == "gpu" else 'cpu')

    video = VideoRecorder(args.work_dir, min_fps=args.min_fps, max_fps=args.max_fps)

    env, eval_env = make_env(args, device)

    # env.map.save_to_disk("Town04")

    video.init(True)

    obs = env.reset(selected_weather=args.selected_weather)

    rgb_dvs_data = {
        "RGB-Frame": [],
        "DVS-Frame": [],
        "DVS-Stream": [],
    }


    for step in range(500):
        print(f"\r\tnow at step: {step}", end="")

        # next_obs, reward, done, _ = env.step(None)
        next_obs, reward, done, _ = env.step([-0.0165, 1.0])    # 高速公路用这个来测试，跑的快点

        for one_key in ["RGB-Frame", "DVS-Frame", "DVS-Stream"]:
            rgb_dvs_data[one_key].append(
                next_obs[one_key].copy()
            )

        # if step <= 125:
        # next_obs, reward, done, _ = env.step([-0.0165, 0.9])    # 方向，油门
        # else:
        #     next_obs, reward, done, _ = env.step([0.2, 0.6])    # 方向，油门

        # if step >= 40:
        #     env.tm.force_lane_change(env.vehicle, True)
        # else:
        #     env.vehicle.set_autopilot(True, env.tm_port)
        #     env.vehicle.set_light_state(carla.VehicleLightState.HighBeam)
        #
        #     env.tm.auto_lane_change(env.vehicle, True)
        #     # next_obs, reward, done, _ = env.step([np.random.uniform(0, 1), 1])
        #     next_obs, reward, done, _ = env.step([0.8, 1])

        video.record(next_obs, None, env.vehicle)

    print("\nall done.")
    video.save(f"{args.domain_name}+{args.selected_weather}")
    #
    # np.savez(f"./test_data/rgb_dvs_data/{args.domain_name}-{args.selected_weather}.npz",
    #          RGB_Frame=rgb_dvs_data["RGB-Frame"],
    #          DVS_Frame=rgb_dvs_data["DVS-Frame"],
    #          DVS_Stream=rgb_dvs_data["DVS-Stream"],
    #          )

if __name__ == '__main__':
    main()



