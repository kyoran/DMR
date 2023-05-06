# !/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from collections import deque


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.obs_shape = obs_shape
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.mtm_length = 10

        self.min_gap_length = 20
        self.max_gap_length = 40


        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32# if len(obs_shape) == 1 else np.uint8
        if isinstance(obs_shape, list) and len(obs_shape) == 2:
            self.rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            self.dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
            self.next_rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            self.next_dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
            # self.k_rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            # self.k_dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
        elif isinstance(obs_shape, list) and len(obs_shape) == 3:
            self.rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            self.dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
            self.depth_obses = np.empty((capacity, *obs_shape[2]), dtype=obs_dtype)
            self.next_rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            self.next_dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
            self.next_depth_obses = np.empty((capacity, *obs_shape[2]), dtype=obs_dtype)
        elif isinstance(obs_shape, list) and len(obs_shape) == 4:
            self.rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            self.dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
            self.depth_obses = np.empty((capacity, *obs_shape[2]), dtype=obs_dtype)
            self.dvs_obses2 = np.empty((capacity, *obs_shape[3]), dtype=obs_dtype)

            self.next_rgb_obses = np.empty((capacity, *obs_shape[0]), dtype=obs_dtype)
            self.next_dvs_obses = np.empty((capacity, *obs_shape[1]), dtype=obs_dtype)
            self.next_depth_obses = np.empty((capacity, *obs_shape[2]), dtype=obs_dtype)
            self.next_dvs_obses2 = np.empty((capacity, *obs_shape[3]), dtype=obs_dtype)

        elif obs_shape[0] == 4:
            self.obses = deque([], maxlen=capacity)        # 里面每个元素是一个numpy数组
            self.next_obses = deque([], maxlen=capacity)
            # self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        else:
            self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.rawidx = 0

    def add(self, obs, action, curr_reward, reward, next_obs, done):

        if isinstance(obs, list) and len(obs) == 2:
            np.copyto(self.rgb_obses[self.idx], obs[0])
            np.copyto(self.dvs_obses[self.idx], obs[1])
            np.copyto(self.next_rgb_obses[self.idx], next_obs[0])
            np.copyto(self.next_dvs_obses[self.idx], next_obs[1])
        elif isinstance(obs, list) and len(obs) == 3:
            np.copyto(self.rgb_obses[self.idx], obs[0])
            np.copyto(self.dvs_obses[self.idx], obs[1])
            np.copyto(self.depth_obses[self.idx], obs[2])
            np.copyto(self.next_rgb_obses[self.idx], next_obs[0])
            np.copyto(self.next_dvs_obses[self.idx], next_obs[1])
            np.copyto(self.next_depth_obses[self.idx], next_obs[2])
        elif isinstance(obs, list) and len(obs) == 4:
            np.copyto(self.rgb_obses[self.idx], obs[0])
            np.copyto(self.dvs_obses[self.idx], obs[1])
            np.copyto(self.depth_obses[self.idx], obs[2])
            np.copyto(self.dvs_obses2[self.idx], obs[3])

            np.copyto(self.next_rgb_obses[self.idx], next_obs[0])
            np.copyto(self.next_dvs_obses[self.idx], next_obs[1])
            np.copyto(self.next_depth_obses[self.idx], next_obs[2])
            np.copyto(self.next_dvs_obses2[self.idx], next_obs[3])

        else:
            if self.obs_shape[0] == 4:  # DVS-Stream单模态
                self.obses.append(obs.copy())
                self.next_obses.append(next_obs.copy())
            else:       # 其他单模态
                np.copyto(self.obses[self.idx], obs)
                np.copyto(self.next_obses[self.idx], next_obs)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, sep=False, multi=False, k=False):

        # idxs = np.random.randint(
        #     0, self.capacity if self.full else self.idx, size=self.batch_size
        # )

        idxs = random.sample(
            np.arange(self.capacity if self.full else self.idx).tolist(),
            self.batch_size
        )

        if sep:
            # idxs = np.random.randint(
            #     0, self.capacity-self.mtm_length if self.full
            #     else self.idx-self.mtm_length, size=self.batch_size
            # )

            # 【method1】完全随机，然后加上步数，但是不完全保证间隔
            # idxs = random.sample(
            #     np.arange(0, self.capacity-self.mtm_length if self.full else self.idx-self.mtm_length).tolist(),
            #     self.batch_size
            # )      # 不重复采样
            # idxs = np.array(idxs)
            # idxs = idxs.reshape(-1, 1)
            # step = np.arange(self.mtm_length, self.mtm_length*2).reshape(1, -1)        #
            # idxs = idxs + step
            # idxs = idxs % self.capacity

            # 【method2】滑动窗口，保证间隔在min_gap和max_gap之间
            curr_idx = np.random.randint(
                0, self.capacity-self.max_gap_length if self.full else self.idx-self.max_gap_length
            )
            idxs = [curr_idx]

            for iii in range(self.batch_size-1):
                pad = np.random.randint(self.min_gap_length, self.max_gap_length)
                idxs.append(
                    curr_idx + pad
                )
                curr_idx = curr_idx + pad
            idxs = np.array(idxs)
            idxs = idxs % self.capacity


            # 【method3】
            # 用behavior similarity来选取负样例
            # selected_idx =
            # selected_idx = np.random.randint(
            #     0, self.capacity if self.full else self.idx, size=self.batch_size
            # )
            # discount = 0.99
            # bisimilarity = r_dist + discount * transition_dist


        if multi:
            rgb_obses = torch.as_tensor(self.rgb_obses[idxs], device=self.device).float()
            dvs_obses = torch.as_tensor(self.dvs_obses[idxs], device=self.device).float()
            # depth_obses = torch.as_tensor(self.depth_obses[idxs], device=self.device).float()
            # dvs_obses2 = torch.as_tensor(self.dvs_obses2[idxs], device=self.device).float()

            next_rgb_obses = torch.as_tensor(self.next_rgb_obses[idxs], device=self.device).float()
            next_dvs_obses = torch.as_tensor(self.next_dvs_obses[idxs], device=self.device).float()
            # next_depth_obses = torch.as_tensor(self.next_depth_obses[idxs], device=self.device).float()
            # next_dvs_obses2 = torch.as_tensor(self.next_dvs_obses2[idxs], device=self.device).float()

            # obses = [rgb_obses, dvs_obses, depth_obses, dvs_obses2]
            obses = [rgb_obses, dvs_obses]
            # next_obses = [next_rgb_obses, next_dvs_obses, next_depth_obses, next_dvs_obses2]
            next_obses = [next_rgb_obses, next_dvs_obses]

        else:
            if self.obs_shape[0] == 4:
                # DVS-stream
                # (batch_size, event_num, 4)

                # 法1：搞成定长，会导致显存爆炸，太大了
                # 统计最大的数量
                # max_event_num = 0
                # for iii in idxs:
                #     if len(self.obses[iii]) > max_event_num:
                #         max_event_num = len(self.obses[iii])
                #     if len(self.next_obses[iii]) > max_event_num:
                #         max_event_num = len(self.next_obses[iii])

                # 补齐数据
                # obses = []
                # next_obses = []
                # for iii in idxs:
                #     # 计算需要补多少
                #     obs_pad_num = max_event_num-len(self.obses[iii])
                #     next_obs_pad_num = max_event_num-len(self.next_obses[iii])
                #     # 提取出来
                #     obs = self.obses[iii]
                #     next_obs = self.next_obses[iii]
                #     # 补齐
                #     # 用0补齐会出现梯度为nan? 所以用1e-6补齐?
                #     if obs_pad_num > 0:
                #         obs = np.pad(obs, ((0, obs_pad_num), (0,0)),
                #                      'constant', constant_values=1e-6)
                #     if next_obs_pad_num > 0:
                #         next_obs = np.pad(self.next_obses[iii], ((0, next_obs_pad_num), (0,0)),
                #                           'constant', constant_values=1e-6)
                #     # 汇总
                #     obses.append(obs)
                #     next_obses.append(next_obs)
                #
                # obses = np.array(obses)
                # next_obses = np.array(next_obses)

                # 返回
                # obses = torch.as_tensor(obses, device=self.device).float()
                # next_obses = torch.as_tensor(next_obses, device=self.device).float()

                # 法2：每个batch里面event事件数量不定
                obses = []
                next_obses = []
                for iii in idxs:
                    tmp_obs = torch.as_tensor(self.obses[iii], device=self.device).float()
                    obses.append(tmp_obs)

                    tmp_next_obs = torch.as_tensor(self.next_obses[iii], device=self.device).float()
                    next_obses.append(tmp_next_obs)

            else:
                # other single modal
                obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
                next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()


        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)

        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs],
                                                                                   device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones


    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)


    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end
