# !/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch

def evaluate(args, image_dir, env, agent, video, num_eval_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    episode_rewards = []
    reward_sum = 0.
    crash_intensity = 0.
    throttle = 0.
    steer = 0.
    brake = 0.
    count = 0

    # embedding visualization
    obses = []
    values = []
    embeddings = []

    for i in range(num_eval_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.

        obs = env.reset()
        #         print('init obs["dvs_events"]', obs["dvs_events"].shape)

        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0

        one_episode_step = 0
        current_perception = None
        while not done:
            #             print('obs["perception"]:', obs["perception"].shape)
            with eval_mode(agent):
                action = agent.select_action(obs["perception"])

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(
                        agent.critic(
                            torch.Tensor(obs).to(device).unsqueeze(0),
                            torch.Tensor(action).to(device).unsqueeze(0)
                        )).item())
                    embeddings.append(
                        agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, done, info = env.step(action)
            one_episode_step += 1

            #             print("now at", one_episode_step, 'obs["dvs_events"]:', obs["dvs_events"].shape,
            #                     "action:", action, "reward:", reward)   # (event_num, 4)
            # metrics:
            if do_carla_metrics:
                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                throttle += info['throttle']
                steer += abs(info['steer'])
                brake += info['brake']
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
            video.record(obs, current_perception, env.vehicle)
            episode_reward += reward

        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)
            reward_sum += episode_reward
            episode_rewards.append(episode_reward)

        video.save(f"{step}")
        L.log('eval/episode_reward', episode_reward, step)

    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print("rewards of each episode: {}".format(episode_rewards))
        print("average_reward: {}".format(reward_sum / num_eval_episodes))
        print("average_distance: {}".format(sum(distance_driven_each_episode) / num_eval_episodes))
        print('crash_intensity: {}'.format(crash_intensity / num_eval_episodes))
        print('throttle: {}'.format(throttle / count))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')


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