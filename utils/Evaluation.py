# !/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import cv2
import torch
import numpy as np


def evaluate(args, image_dir, eval_env, agent, video, num_eval_episodes, L, step,
             one_weather, device=None, embed_viz_dir=None, do_metrics=None, suit='carla'):

    print("@"*50 + f"at step: [{step}]")
    # L.log('eval/episode', episode, step)

    # for one_weather in [
    #     # "hard_high_light",
    #     # "soft_high_light",
    #     # "soft_low_light",
    #     "hard_low_light",
    #     # "soft_noisy_low_light",
    #     # "hard_noisy_low_light",
    # ]:
    # # for one_weather in [
    # #     "hard_noisy_low_light",
    # #     "hard_low_light",
    # #     "hard_high_light",
    # # ]:
    #     # L.log('eval/episode', episode, step)
    print("#" * 50)
    # print(f"Now eval env at {one_weather} ...")

    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity_each_episode = []
    reward_each_episode = []

    throttle_each_episode = []
    steer_each_episode = []
    brake_each_episode = []
    reward_sum = 0.
    crash_intensity = 0.
    throttle = 0.
    steer = 0.
    brake = 0.
    count = 0

    # embedding visualization
    rgb_obses = []
    dvs_obses = []
    values = []
    rgb_convs = []
    dvs_convs = []
    con_convs = []
    rgb_embeddings = []
    dvs_embeddings = []
    con_embeddings = []

    for i in range(num_eval_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.
        episode_reward = 0.
        crash_intensity_in_each_episode = 0.
        throttle_in_each_episode = 0.
        steer_in_each_episode = 0.
        brake_in_each_episode = 0.

        if suit == 'carla':
            obs = eval_env.reset(selected_weather=one_weather)
        else:
            obs = eval_env.reset()
        # obs = eval_env.reset()
        #         print('init obs["dvs_events"]', obs["dvs_events"].shape)

        video.init(enabled=(i == 0))
        done = False

        one_episode_step = 0
        current_perception = None

        while not done:
            # print(f"now at: {i}-th episode, {one_episode_step}-th step")
            # print('obs["perception"]:', obs["perception"].shape)
            with eval_mode(agent):
                # print('obs["perception"]:', obs["perception"].shape)
                action = agent.select_action(obs["perception"])

            if True and embed_viz_dir and step >= 100000:
                rgb_obses.append(obs["perception"][0][3:6].copy())
                dvs_obses.append(obs["DVS-Frame"].copy())

                with torch.no_grad():
                    # values.append(min(
                    #     agent.critic(
                    #         [
                    #             torch.Tensor(obs["perception"][0]).unsqueeze(0).to(device),
                    #             torch.Tensor(obs["perception"][1]).unsqueeze(0).to(device),
                    #         ],
                    #         torch.Tensor(action).to(device).unsqueeze(0)
                    #     )).item())

                    # fuse_h, [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv] \
                    con_h, [rgb_h, con_h, dvs_h], con_conv, [rgb_conv, dvs_conv] \
                        = agent.critic.encoder(
                        [
                            torch.Tensor(obs["perception"][0]).unsqueeze(0).to(device),
                            torch.Tensor(obs["perception"][1]).unsqueeze(0).to(device),
                        ], vis=True
                    )

                    rgb_h = rgb_h.cpu().detach().numpy()
                    dvs_h = dvs_h.cpu().detach().numpy()
                    con_h = con_h.cpu().detach().numpy()
                    rgb_embeddings.append(rgb_h)
                    dvs_embeddings.append(dvs_h)
                    con_embeddings.append(con_h)

                    rgb_conv = rgb_conv.cpu().squeeze().detach().numpy()
                    dvs_conv = dvs_conv.cpu().squeeze().detach().numpy()
                    con_conv = con_conv.cpu().squeeze().detach().numpy()
                    rgb_convs.append(np.sum(rgb_conv, axis=0))
                    dvs_convs.append(np.sum(dvs_conv, axis=0))
                    con_convs.append(np.sum(con_conv, axis=0))

            obs, reward, done, info = eval_env.step(action)
            one_episode_step += 1

            #             print("now at", one_episode_step, 'obs["dvs_events"]:', obs["dvs_events"].shape,
            #                     "action:", action, "reward:", reward)   # (event_num, 4)
            # metrics:
            if do_metrics:
                if suit == 'carla':
                    dist_driven_this_episode += info['distance']
                    crash_intensity += info['crash_intensity']
                    crash_intensity_in_each_episode += info['crash_intensity']
                    episode_reward += reward

                    throttle += info['throttle']
                    throttle_in_each_episode += info['throttle']
                    steer += abs(info['steer'])
                    steer_in_each_episode += abs(info['steer'])
                    brake += info['brake']
                    brake_in_each_episode += info['brake']
                    count += 1

                elif suit == 'airsim':
                    # print("@@@@info:", info)
                    dist_driven_this_episode += info['distance']
                    episode_reward += reward

                    # throttle += info['throttle']
                    # throttle_in_each_episode += info['throttle']
                    # steer += abs(info['steer'])
                    # steer_in_each_episode += abs(info['steer'])
                    # brake += info['brake']
                    # brake_in_each_episode += info['brake']
                    count += 1

            if False and i == 0:
                """
                obs[third_person_rgb].shape: (600, 800, 3)
                obs[dvs_frame].shape: (3, 84, 420)
                obs[rgb_frame].shape: (84, 420, 3)
                self.third_person_rgb_frames.append(obs["third_person_rgb"])
                self.dvs_frames.append(np.transpose(obs["dvs_frame"], [1, 2, 0]))
                self.rgb_frames.append(obs["rgb_frame"])
                """
                cv2.imwrite(
                    os.path.join(image_dir, f'rgb_frame-{i}-{one_episode_step}.jpg'),
                    cv2.cvtColor(obs["rgb_frame"], cv2.COLOR_RGB2BGR)
                )

            # video.record(obs)
            if suit == 'carla':
                video.record(obs, current_perception, eval_env.vehicle)
            elif suit == 'airsim':
                video.record(obs, current_perception, None)

        if 'carla_bug' in info['reason_episode_ended']:
            print('carla bug, skip this episode.')
            continue

        # metrics:
        if do_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)
            crash_intensity_each_episode.append(crash_intensity_in_each_episode)
            # throttle_each_episode.append(throttle / count)
            # steer_each_episode.append(steer / count)
            # brake_each_episode.append(brake / count)
            reward_each_episode.append(episode_reward)
            reward_sum += episode_reward

        if suit == 'carla':
            video.save(f"{eval_env.selected_scenario}-{eval_env.selected_weather}-{step}")
            L.log(f'eval/{eval_env.selected_scenario}/{eval_env.selected_weather}/episode_reward', episode_reward, step)
            L.log(f'eval/{eval_env.selected_scenario}/{eval_env.selected_weather}/episode_dist', sum(distance_driven_each_episode) / num_eval_episodes, step)
            L.log(f'eval/{eval_env.selected_scenario}/{eval_env.selected_weather}/episode_crash', crash_intensity / num_eval_episodes, step)
            L.log(f'eval/{eval_env.selected_scenario}/{eval_env.selected_weather}/episode_throttle', throttle / count, step)
            L.log(f'eval/{eval_env.selected_scenario}/{eval_env.selected_weather}/episode_steer', steer / count, step)
            L.log(f'eval/{eval_env.selected_scenario}/{eval_env.selected_weather}/episode_brake', brake / count, step)
        elif suit == 'airsim':
            video.save(f"airsim-{step}")
            L.log(f'eval/airsim/episode_reward', episode_reward, step)
            L.log(f'eval/airsim/episode_dist', sum(distance_driven_each_episode) / num_eval_episodes, step)


    if embed_viz_dir:
        dataset = {
            'rgb_obses': rgb_obses, 'dvs_obses': dvs_obses,
            # 'values': values,
            'rgb_convs': rgb_convs, 'dvs_convs': dvs_convs, 'con_convs': con_convs,
            'rgb_embeddings': rgb_embeddings, 'dvs_embeddings': dvs_embeddings, 'con_embeddings': con_embeddings
        }
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    if do_metrics:
        if suit == 'carla':
            print('---------------------------------')
            print(f'METRICS of {eval_env.selected_scenario}-{eval_env.selected_weather}')
            print("reason_each_episode_ended: {}".format(reason_each_episode_ended))

            print("distance_driven_each_episode: {}".format(distance_driven_each_episode))  # 每轮中所有步数的行驶距离和
            print("->average_distance: {}".format(sum(distance_driven_each_episode) / num_eval_episodes))

            print("crash_intensity_each_episode: {}".format(crash_intensity_each_episode))
            print('->average_crash_intensity: {}'.format(crash_intensity / num_eval_episodes))

            print("reward_each_episode: {}".format(reward_each_episode))       # 每轮中所有步数的奖励和
            print("->average_reward: {}".format(reward_sum / num_eval_episodes))

            # print("throttle_each_episode: {}".format(throttle_each_episode))    # 按步数平均的油门
            # print("steer_each_episode: {}".format(steer_each_episode))          # 按步数平均的方向
            # print("brake_each_episode: {}".format(brake_each_episode))          # 按步数平均的刹车
            #
            print('throttle: {}'.format(throttle / count))
            print('steer: {}'.format(steer / count))
            print('brake: {}'.format(brake / count))
            print("#" * 50)

        elif suit == 'airsim':
            print('---------------------------------')
            print(f'METRICS of airsim')
            print("reason_each_episode_ended: {}".format(reason_each_episode_ended))

            print("distance_driven_each_episode: {}".format(distance_driven_each_episode))  # 每轮中所有步数的行驶距离和
            print("->average_distance: {}".format(sum(distance_driven_each_episode) / num_eval_episodes))

            print("reward_each_episode: {}".format(reward_each_episode))  # 每轮中所有步数的奖励和
            print("->average_reward: {}".format(reward_sum / num_eval_episodes))

            print("#" * 50)
        # L.log(f'eval/{eval_env.selected_weather}/avg_reward', reward_sum / num_eval_episodes, step)
        # L.log(f'eval/{eval_env.selected_weather}/avg_distance', sum(distance_driven_each_episode) / num_eval_episodes, step)

    # L.logs(f'eval/metrics-{eval_env.selected_weather}', {
    #     "avg_distance": sum(distance_driven_each_episode) / num_eval_episodes,
    #     "avg_reward": sum(distance_driven_each_episode) / num_eval_episodes
    # }, step)

    L.dump(step)

    print("@"*50)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False