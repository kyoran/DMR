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
import matplotlib
import matplotlib.pyplot as plt
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
from utils.Evaluation1 import eval_mode, evaluate
from utils.seed_everywhere import seed_everywhere


from encoder import pixelInputFusion as InputFuse_ENCODER
from encoder import pixelCrossFusion as CrossFuse_ENCODER
from encoder import pixelCat as Late_ENCODER
from encoder import pixelCon as OURS_ENCODER

hd_height, hd_width = 512, 512
# hd_height, hd_width = 256, 256

hd_rgb_data = {'img': np.zeros((hd_height, hd_width, 3), dtype=np.uint8)}
hd_dvs_data = {'img': np.zeros((hd_height, hd_width, 3), dtype=np.uint8)}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--suit', default='carla', choices=['carla', 'airsim'])
    ###########################---scenario---###############################
    parser.add_argument('--load_model_step', default=109999)
    # parser.add_argument('--domain_name', default='normal')

    parser.add_argument('--domain_name', default='highbeam')
    # parser.add_argument('--domain_name', default='tunnel')

    # parser.add_argument('--domain_name', default='highspeed')
    # parser.add_argument('--domain_name', default='superspeed')
    #
    # parser.add_argument('--domain_name', default='jaywalk')
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
    parser.add_argument('--carla_rpc_port', default=12121, type=int)
    parser.add_argument('--carla_tm_port', default=19121, type=int)
    parser.add_argument('--carla_timeout', default=120, type=int)
    # environment
    # parser.add_argument('--work_dir', default=fr'E:\data\xuhr\nips2023\seperator', type=str)
    parser.add_argument('--work_dir', default=fr'E:\logs\final', type=str)
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

    # parser.add_argument('--SEED', default=111, type=int)
    # parser.add_argument('--SEED', default=222, type=int)
    parser.add_argument('--SEED', default=333, type=int)

    parser.add_argument('--do_carla_metrics', default=True)
    parser.add_argument('--LOG_FREQ', default=10000, type=int)
    parser.add_argument('--EVAL_FREQ', default=50, type=int)
    parser.add_argument('--SAVE_MODEL_FREQ', default=20, type=int)
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    parser.add_argument('--min_stuck_steps', default=100, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--fov', default=60, type=int)
    parser.add_argument('--rl_image_size', default=128, type=int)
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
    # parser.add_argument('--encoder_type', default='pixelConNewV4', type=str)
    parser.add_argument('--encoder_type', default='pixelCrossFusion', type=str)
    # parser.add_argument('--encoder_type', default='pixelEFNet', type=str)
    # parser.add_argument('--encoder_type', default='pixelFPNNet', type=str)
    # parser.add_argument('--encoder_type', default='pixelRENet', type=str)
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

    args.work_dir = args.work_dir + rf"\carla+{args.domain_name}+{args.selected_weather}+{args.agent}+{args.perception_type}+{args.encoder_type}+{args.SEED}"

    return args


def make_env(args, device):
    if args.suit == 'carla':
        with open('../env/CARLA_0_9_13_pithy/weather.json', 'r', encoding='utf8') as fff:
            weather_params = json.load(fff)
        with open('../env/CARLA_0_9_13_pithy/scenario.json', 'r', encoding='utf8') as fff:
            scenario_params = json.load(fff)
        with open('../tools/rpg_e2vid/dvs_rec_args.json', 'r', encoding='utf8') as fff:
            dvs_rec_args = json.load(fff)
            dvs_rec_args = dotdict(dvs_rec_args)

        env = CarlaEnv(
            weather_params=weather_params,
            scenario_params=scenario_params,
            dvs_rec_args=dvs_rec_args,
            selected_scenario=args.domain_name,
            selected_weather=args.selected_weather,
            # selected_weather="cloudy",
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


def calc_conv(args, conv, size=None):
    """
    https://blog.csdn.net/wsp_1138886114/article/details/118552328
    """
    conv = conv.squeeze().cpu().detach().numpy()
    conv = np.sum(conv, axis=0)  # (6, 6)
    if size is None:
        conv = cv2.resize(conv, (args.rl_image_size, args.rl_image_size))  # (128, 128)
    else:
        conv = cv2.resize(conv, (size[0], size[1]))  # (1024, 1024)

    conv = np.uint8((conv - np.min(conv)) / (np.max(conv) - np.min(conv)) * 255)
    # conv = cv2.applyColorMap(conv, cv2.COLORMAP_HSV)
    # conv = cv2.applyColorMap(conv, cv2.COLORMAP_VIRIDIS)    # 翠绿色
    conv = cv2.applyColorMap(conv, cv2.COLORMAP_JET)    #
    conv = cv2.cvtColor(conv, cv2.COLOR_BGR2RGB)
    return conv

def main():
    args = parse_args()

    seed_everywhere(args.SEED)
    device = torch.device(f'cuda' if torch.cuda.is_available() and args.device == "gpu" else 'cpu')

    # video = VideoRecorder(args.work_dir, min_fps=args.min_fps, max_fps=args.max_fps)
    # print("@@@@@@@@@@@@@ready to make env ...")

    env, eval_env = make_env(args, device)

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device,
    )
    agent.load(os.path.join(args.work_dir, 'model'), args.load_model_step)


    # env.map.save_to_disk("Town04")

    # video.init(True)


    obs = env.reset(
        selected_weather=args.selected_weather
    )

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 生成高清的图像，用于可视化
    hd_rgb_camera_bp = env.bp_lib.find('sensor.camera.rgb')
    hd_rgb_camera_bp.set_attribute('sensor_tick', f'{1 / env.min_fps}')
    hd_rgb_camera_bp.set_attribute('image_size_x', str(hd_width))
    hd_rgb_camera_bp.set_attribute('image_size_y', str(hd_height))
    hd_rgb_camera_bp.set_attribute('fov', str(env.fov))
    hd_rgb_camera_bp.set_attribute('enable_postprocess_effects', str(True))  # a set of post-process effects is applied to the image to create a more realistic feel
    hd_rgb_camera_bp.set_attribute('exposure_max_bright', '20.0')  # over-exposure
    hd_rgb_camera_bp.set_attribute('exposure_min_bright', '11.0')  # under-exposure，默认是10，越小越亮
    hd_rgb_camera_bp.set_attribute('blur_amount', '1.0')
    hd_rgb_camera_bp.set_attribute('motion_blur_intensity', '1.0')
    hd_rgb_camera_bp.set_attribute('motion_blur_max_distortion', '0.8')
    hd_rgb_camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.4')
    hd_rgb_camera_bp.set_attribute('exposure_speed_up', '3.0')  # Speed at which the adaptation occurs from dark to bright environment.
    hd_rgb_camera_bp.set_attribute('exposure_speed_down', '1.0')  # Speed at which the adaptation occurs from bright to dark environment.
    hd_rgb_camera_bp.set_attribute('lens_flare_intensity', '0.2')  # Intensity for the lens flare post-process effect （光晕效果）
    hd_rgb_camera_bp.set_attribute('shutter_speed', '100')  # The camera shutter speed in seconds 快门速度

    hd_dvs_camera_bp = env.bp_lib.find('sensor.camera.dvs')
    hd_dvs_camera_bp.set_attribute('sensor_tick', f'{1 / env.max_fps}')
    hd_dvs_camera_bp.set_attribute('positive_threshold', str(0.15))  # 光强变化阈值，0.3是默认值，下雨几乎没噪声，0.2时雨点太少，0.1时雨点太多
    hd_dvs_camera_bp.set_attribute('negative_threshold', str(0.15))
    hd_dvs_camera_bp.set_attribute('sigma_positive_threshold', str(0.01))  # 白噪声（摄像头自身电器原件噪声）
    hd_dvs_camera_bp.set_attribute('sigma_negative_threshold', str(0.01))
    hd_dvs_camera_bp.set_attribute('image_size_x', str(hd_width))
    hd_dvs_camera_bp.set_attribute('image_size_y', str(hd_height))
    hd_dvs_camera_bp.set_attribute('use_log', str(True))  # 不用log，变化太大，噪声更多，log平滑曲线，要用log！
    hd_dvs_camera_bp.set_attribute('fov', str(env.fov))
    hd_dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))


    location = carla.Location(x=1, z=1.5)

    def __get_hd_rgb_data__(data):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        hd_rgb_data['frame'] = data.frame
        hd_rgb_data['timestamp'] = data.timestamp
        hd_rgb_data['img'] = array
    hd_rgb_camera = env.world.spawn_actor(
        hd_rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
        attach_to=env.vehicle)
    hd_rgb_camera.listen(lambda data: __get_hd_rgb_data__(data))
    env.sensor_actors.append(hd_rgb_camera)

    def __get_hd_dvs_data__(data):
        events = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)]))
        hd_dvs_data['frame'] = data.frame
        hd_dvs_data['timestamp'] = data.timestamp
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        p = events['pol'].astype(np.float32)
        t = events['t'].astype(np.float32)
        events = np.column_stack((x, y, p, t))  # (event_num, 4)
        hd_dvs_data['events'] = events
        hd_dvs_data['events'] = hd_dvs_data['events'][np.argsort(hd_dvs_data['events'][:, -1])]
        hd_dvs_data['events'] = hd_dvs_data['events'].astype(np.float32)
        img = np.zeros((hd_height, hd_width, 3), dtype=np.uint8)      # 0是黑色
        # img = np.ones((hd_height, hd_width, 3), dtype=np.uint8) * 255  # 255 是白色，不能这样，会导致蓝红看不见了
        img[hd_dvs_data['events'][:, 1].astype(np.int),
        hd_dvs_data['events'][:, 0].astype(np.int),
        hd_dvs_data['events'][:, 2].astype(np.int) * 2] = 255
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 没有dvs事件的位置，将黑色改为白色，有dvs事件的地方保留黑色是为了让蓝红显示出来
        non_dvs_indx = np.logical_and(img[:, :, 0] != 255, img[:, :, 2] != 255)
        img[non_dvs_indx] = 255
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        hd_dvs_data['img'] = img

    hd_dvs_camera = env.world.spawn_actor(
        hd_dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
        attach_to=env.vehicle)
    hd_dvs_camera.listen(lambda data: __get_hd_dvs_data__(data))
    env.sensor_actors.append(hd_dvs_camera)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    # rgb_dvs_data = {
    #     "RGB-Frame": [],
    #     "DVS-Frame": [],
    #     "DVS-Stream": [],
    # }
    matplotlib.use('agg')
    # matplotlib.use('TkAgg')
    alpha_rgb = 0.5     #!!!!!!!!!!!!!!!!!!!!!!!
    alpha_dvs = 0.6     #!!!!!!!!!!!!!!!!!!!!!!!
    plt.ion()
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for iii in range(3):
        for jjj in range(3):
            ax[iii, jjj].axis('off')



    for step in range(1000):
        # plt.pause(0.0001)

        print(f"\r\tnow at step: {step}", end="")

        # next_obs, reward, done, _ = env.step(None)
        # next_obs, reward, done, _ = env.step([-0.0165, 1.0])    # 高速公路用这个来测试，跑的快点
        next_obs, reward, done, _ = env.step([0, 1.0])    # 高速公路用这个来测试，跑的快点
        # print("hd_rgb_data['img']:", hd_rgb_data['img'].shape)  # (1024, 1024, 3)
        # print("hd_dvs_data['img']:", hd_dvs_data['img'].shape)  # (1024, 1024, 3)

        if step <= 440:
            continue

        # print("next_obs:", next_obs["DVS-Frame"].shape)     # (128, 128, 3)
        # print("next_obs:", next_obs["RGB-Frame"].shape)     # (128, 128, 3)
        # print("RGB:", next_obs["RGB-Frame"].shape, next_obs["RGB-Frame"].max(), next_obs["RGB-Frame"].min())
        ax[0, 0].imshow(hd_rgb_data['img'])
        ax[0, 0].set_title("RGB-Frame")
        ax[0, 1].imshow(hd_dvs_data['img'])
        ax[0, 1].set_title("DVS-Frame")

        # print("rgb.shape:", next_obs["perception"][0].shape)    # (9, 128, 128)
        # print("dvs.shape:", next_obs["perception"][1].shape)    # (15, 128, 128)
        input_obs = [
            torch.Tensor(next_obs["perception"][0]).unsqueeze(0).to(device),
            torch.Tensor(next_obs["perception"][1]).unsqueeze(0).to(device),
        ]

        if args.encoder_type == "pixelCrossFusion":
            _, [_, _], [rgb_conv, dvs_conv] = agent.critic.encoder(input_obs, vis=True)

            rgb_conv = calc_conv(args, rgb_conv, [hd_height, hd_width])
            dvs_conv = calc_conv(args, dvs_conv, [hd_height, hd_width])
            ax[1, 0].imshow(rgb_conv)
            ax[1, 0].set_title("rgb_conv")
            ax[1, 1].imshow(dvs_conv)
            ax[1, 1].set_title("dvs_conv")

            superimposed_rgb = rgb_conv * alpha_rgb + hd_rgb_data['img'] * (1 - alpha_rgb)
            superimposed_rgb = np.uint8(superimposed_rgb)
            superimposed_dvs = dvs_conv * alpha_dvs + hd_dvs_data['img'] * (1 - alpha_dvs)
            superimposed_dvs = np.uint8(superimposed_dvs)
            ax[2, 0].imshow(superimposed_rgb)
            ax[2, 0].set_title("rgb")
            ax[2, 1].imshow(superimposed_dvs)
            ax[2, 1].set_title("dvs")

        elif args.encoder_type == "pixelEFNet" or args.encoder_type == "pixelFPNNet" or args.encoder_type == "pixelRENet":
            _, _, fused_conv = agent.critic.encoder(input_obs, vis=True)
            print("fused_conv:", fused_conv.shape)
            fused_conv = calc_conv(args, fused_conv, [hd_height, hd_width])
            ax[1, 0].imshow(fused_conv)
            ax[1, 0].set_title("fused_conv")

            superimposed_rgb = fused_conv * alpha_rgb + hd_rgb_data['img'] * (1 - alpha_rgb)
            superimposed_rgb = np.uint8(superimposed_rgb)
            superimposed_dvs = fused_conv * alpha_dvs + hd_dvs_data['img'] * (1 - alpha_dvs)
            superimposed_dvs = np.uint8(superimposed_dvs)
            ax[2, 0].imshow(superimposed_rgb)
            ax[2, 0].set_title("rgb")
            ax[2, 1].imshow(superimposed_dvs)
            ax[2, 1].set_title("dvs")

        saved_dir = f"./conv_{args.domain_name}_{args.selected_weather}_{args.encoder_type}"

        rgb_extent = ax[0,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        dvs_extent = ax[0,1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        rgb_mask_extent = ax[1,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        dvs_mask_extent = ax[1,1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        rgb_mask_sup_extent = ax[2,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        dvs_mask_sup_extent = ax[2,1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(f"{saved_dir}/{step}-rgb.jpg", bbox_inches=rgb_extent)
        plt.savefig(f"{saved_dir}/{step}-dvs.jpg", bbox_inches=dvs_extent)
        plt.savefig(f"{saved_dir}/{step}-rgb-mask.jpg", bbox_inches=rgb_mask_extent)
        plt.savefig(f"{saved_dir}/{step}-dvs-mask.jpg", bbox_inches=dvs_mask_extent)
        plt.savefig(f"{saved_dir}/{step}-rgb-mask-sup.jpg", bbox_inches=rgb_mask_sup_extent)
        plt.savefig(f"{saved_dir}/{step}-dvs-mask-sup.jpg", bbox_inches=dvs_mask_sup_extent)

        plt.savefig(f"{saved_dir}/{step}-conv.jpg")

        # video.record(next_obs, None, env.vehicle)

    print("\nall done.")
    # video.save(f"{args.domain_name}+{args.selected_weather}")

    # np.savez(f"./test_data/rgb_dvs_data/{args.domain_name}-{args.selected_weather}.npz",
    #          RGB_Frame=rgb_dvs_data["RGB-Frame"],
    #          DVS_Frame=rgb_dvs_data["DVS-Frame"],
    #          DVS_Stream=rgb_dvs_data["DVS-Stream"],
    #          )

if __name__ == '__main__':
    main()



