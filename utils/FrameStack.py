# !/usr/bin/python3
# -*- coding: utf-8 -*-

import gym
import copy
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):

    def __init__(self, env, k, DENOISE, type="RGB-Frame"):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._carousel = deque([], maxlen=k)
        self.DENOISE = DENOISE
        self.type = type

        shp = env.observation_space.shape
        # import pdb; pdb.set_trace()

        if isinstance(shp, list) and len(shp) >= 2:
            # Multi-Modals
            stack_shp = []
            for one_shp in shp:
                if one_shp[1] == 4:     # DVS-Stream的四元祖
                    stack_shp.append((4,))
                else:
                    stack_shp.append(
                        (one_shp[0] * k,) + one_shp[1:]
                    )

            self.observation_space.shape = stack_shp.copy()

        else:

            if shp[1] == 4:     # DVS-Stream的四元祖
                self.observation_space.shape = (4,)
            else:
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=((shp[0] * k,) + shp[1:]),
                    # shape=((shp[0][0] * k+shp[1][0]*k,) + shp[0][1:]+shp[1][1:]),
                    dtype=env.observation_space.dtype
                )

    def _get_perception(self, obs):

        if self.perception_type.__contains__("+"):
            # 多模态
            modals = self.perception_type.split("+")

        else:
            modals = [self.perception_type]

        perception = []

        for one_modal in modals:

            if one_modal == "RGB-Frame":
                perception.append(obs[one_modal])

            elif one_modal == "DVS-Stream":
                perception.append(obs[one_modal])

            elif one_modal == "DVS-Frame":
                perception.append(obs[one_modal][:, :, [0, 2]])

            elif one_modal == "DVS-Voxel-Grid":
                perception.append(obs[one_modal])

            elif one_modal == "E2VID-Frame":
                perception.append(obs[one_modal])

            elif one_modal == "Depth-Frame":
                perception.append(obs[one_modal])

            elif one_modal == "LiDAR-BEV":
                perception.append(obs[one_modal])

            elif one_modal == "LiDAR-PCD":
                perception.append(obs[one_modal])

        if len(modals) == 1:
            return perception[0]
        else:
            return perception


    def reset(self, selected_weather=None):
        # print("in reset")
        obs = self.env.reset(selected_weather=selected_weather)
        # print("reset done")

        for _ in range(self._k):
            self._carousel.append(self._get_perception(obs))

        # print("!!!:", len(self._carousel))
        stack_perception = self._get_stack_perception()
        obs.update({
            'perception': stack_perception
        })
        return obs


    def step(self, action):
        #         obs, reward, done, info = self.env.step(action)
        #         self._frames.append(obs)
        #         return self._get_obs(), reward, done, info
        obs, reward, done, info = self.env.step(action)
        self._carousel.append(self._get_perception(obs))

        stack_perception = self._get_stack_perception()

        obs.update({
            'perception': stack_perception
        })
        return obs, reward, done, info


    def _get_stack_perception(self):


        if self.perception_type.__contains__("+"):
            # 多模态
            modals = self.perception_type.split("+")

            stack_perception = []     # 每个模态一个列表

            for one_modal_idx in range(len(modals)):

                # 分别stack每个模态的数据
                one_modal_stack_perception = []  # 某个模态的stack起来

                for one_k in range(self._k):

                    # stream要单独stack，因为操作不一样
                    if modals[one_modal_idx] == "DVS-Stream":
                        tmp_perception = np.concatenate(self._carousel[one_k][one_modal_idx], axis=1).squeeze()

                        # (x, y, p, t)
                        tmp_perception = tmp_perception[
                            np.argsort(tmp_perception[:, -1])[::-1]]  # times is like: [1, ...., 0]
                        one_modal_stack_perception.append(tmp_perception)

                    elif modals[one_modal_idx] == "LiDAR-PCD":
                        tmp_perception = np.concatenate(self._carousel[one_k][one_modal_idx], axis=1).squeeze()
                        # (x, y, z, d)
                        one_modal_stack_perception.append(tmp_perception)

                    else:
                        # print(len(self._carousel))
                        # print("@@@:", len(self._carousel[0]))
                        # print("@@@:", len(self._carousel[1]))
                        # print("@@@:", len(self._carousel[2]))
                        one_modal_stack_perception.append(self._carousel[one_k][one_modal_idx])

                # stack结束，放到总列表里面返回
                stack_perception.append(
                    np.transpose(
                        np.concatenate(one_modal_stack_perception, axis=2),
                        (2, 0, 1)
                    ).astype(np.float32)
                )

            return stack_perception
        else:
            # 单模态
            if self.perception_type == "DVS-Stream":
                # print(self._carousel)
                # print("self._carousel:", self._carousel[0].shape, self._carousel[1].shape, self._carousel[2].shape)
                stack_perception = np.concatenate(self._carousel, axis=0).squeeze()
                stack_perception = stack_perception[np.argsort(stack_perception[:, -1])[::-1]]  # times is like: [1, ...., 0]

                return stack_perception

            else:

                return np.transpose(
                        np.concatenate(self._carousel, axis=2),
                        (2, 0, 1)
                    ).astype(np.float32)




