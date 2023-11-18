# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

"""
sudo docker run --privileged --user carla --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla-add:0.9.13 /bin/bash ./CarlaUE4.sh -world-port=12321 -RenderOffScreen
"""

import carla

import os
import cv2
import sys
import time
import math
import torch
import random
import numpy as np
from copy import deepcopy
from dotmap import DotMap


class CarlaEnv(object):

    def __init__(self,
                 weather_params, scenario_params,
                 selected_weather, selected_scenario,
                 carla_rpc_port, carla_tm_port, carla_timeout,
                 perception_type, num_cameras, rl_image_size,
                 fov, max_fps, min_fps, device,
                 min_stuck_steps, max_episode_steps, frame_skip,
                 DENOISE=False, is_spectator=False, ego_auto_pilot=False,
                 TPV=False, BEV=False
                 ):

        self.device = device
        self.DENOISE = DENOISE
        self.TPV = TPV
        self.BEV = BEV
        self.frame_skip = frame_skip

        self.carla_rpc_port = carla_rpc_port
        self.carla_tm_port = carla_tm_port
        self.carla_timeout = carla_timeout
        self.weather_params = weather_params
        self.scenario_params = scenario_params

        # testing params
        self.ego_auto_pilot = ego_auto_pilot
        self.is_spectator = is_spectator

        self.num_cameras = num_cameras
        self.rl_image_size = rl_image_size
        self.fov = fov
        self.max_fps = max_fps
        self.min_fps = min_fps
        if max_episode_steps is None:
            self.max_episode_steps = 20 * self.max_fps
        else:
            self.max_episode_steps = max_episode_steps
        if min_stuck_steps is None:
            self.min_stuck_steps = 2 * self.max_fps
        else:
            self.min_stuck_steps = min_stuck_steps

        self.selected_weather = selected_weather
        self.selected_scenario = selected_scenario

        # rgb-frame, dvs-rec-frame, dvs-stream, dvs-vidar-stream
        self.perception_type = perception_type  # ↑↑↑↑↑↑↑↑↑

        # client init
        self.client = carla.Client('localhost', self.carla_rpc_port)
        self.client.set_timeout(self.carla_timeout)

        # world
        self.world = self.client.load_world(self.scenario_params[self.selected_scenario]["map"])

        # assert self.client.get_client_version() == "0.9.13"
        assert self.selected_scenario in self.scenario_params.keys()
        assert self.selected_weather in self.weather_params.keys()

        #
        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_ai_actors = []
        self.walker_actors = []

        self.reset_num = 0

        # reset
        self.reset()


    def _init_blueprints(self):

        self.bp_lib = self.world.get_blueprint_library()

        self.collision_bp = self.bp_lib.find('sensor.other.collision')

        self.bev_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.bev_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.bev_camera_bp.set_attribute('image_size_x', str(2048))
        self.bev_camera_bp.set_attribute('image_size_y', str(2048))
        self.bev_camera_bp.set_attribute('fov', str(90))
        self.bev_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        self.video_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.video_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.video_camera_bp.set_attribute('image_size_x', str(1024))
        self.video_camera_bp.set_attribute('image_size_y', str(1024))


        self.rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.rgb_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.rgb_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.rgb_camera_bp.set_attribute('fov', str(self.fov))
        self.rgb_camera_bp.set_attribute('enable_postprocess_effects', str(True))   # a set of post-process effects is applied to the image to create a more realistic feel
        self.rgb_camera_bp.set_attribute('exposure_max_bright', '20.0')      # over-exposure
        self.rgb_camera_bp.set_attribute('exposure_min_bright', '12.0')      # under-exposure，默认是10，越小越亮
        # self.rgb_camera_bp.set_attribute('exposure_min_bright', '9')      # under-exposure，默认是10，越小越亮
        self.rgb_camera_bp.set_attribute('blur_amount', '1.0')
        self.rgb_camera_bp.set_attribute('motion_blur_intensity', '1.0')
        self.rgb_camera_bp.set_attribute('motion_blur_max_distortion', '0.8')
        self.rgb_camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.4')
        self.rgb_camera_bp.set_attribute('exposure_speed_up', '3.0')    # Speed at which the adaptation occurs from dark to bright environment.
        self.rgb_camera_bp.set_attribute('exposure_speed_down', '1.0')  # Speed at which the adaptation occurs from bright to dark environment.
        self.rgb_camera_bp.set_attribute('lens_flare_intensity', '0.2')  # 	Intensity for the lens flare post-process effect （光晕效果）
        self.rgb_camera_bp.set_attribute('shutter_speed', '100')  # The camera shutter speed in seconds 快门速度



        self.dvs_camera_bp = self.bp_lib.find('sensor.camera.dvs')
        self.dvs_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        self.dvs_camera_bp.set_attribute('positive_threshold', str(0.15))      # 光强变化阈值，0.3是默认值，下雨几乎没噪声，0.2时雨点太少，0.1时雨点太多
        self.dvs_camera_bp.set_attribute('negative_threshold', str(0.15))
        self.dvs_camera_bp.set_attribute('sigma_positive_threshold', str(0.1))  # noise
        self.dvs_camera_bp.set_attribute('sigma_negative_threshold', str(0.1))
        self.dvs_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.dvs_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        # self.dvs_camera_bp.set_attribute('use_log', str(False))
        self.dvs_camera_bp.set_attribute('use_log', str(True))  # using log is more reasonable in reality
        self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))


        self.depth_camera_bp = self.bp_lib.find('sensor.camera.depth')
        self.depth_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.depth_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        self.lidar_camera_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_channel = 64
        self.lidar_camera_bp.set_attribute('channels', str(self.lidar_channel))
        self.lidar_range = 100
        self.lidar_camera_bp.set_attribute('range', str(self.lidar_range))
        self.lidar_camera_bp.set_attribute('points_per_second', '250000')
        self.lidar_camera_bp.set_attribute('rotation_frequency', '20')
        self.lidar_camera_bp.set_attribute('dropoff_general_rate',
                    self.lidar_camera_bp.get_attribute('dropoff_general_rate').recommended_values[0])
        self.lidar_camera_bp.set_attribute('dropoff_intensity_limit',
                    self.lidar_camera_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        self.lidar_camera_bp.set_attribute('dropoff_zero_intensity',
                    self.lidar_camera_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

        # self.vidar_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        # self.vidar_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        # self.vidar_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        # self.vidar_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        # self.vidar_camera_bp.set_attribute('fov', str(self.fov))
        # self.vidar_camera_bp.set_attribute('enable_postprocess_effects', str(True))

    def _set_dummy_variables(self):
        # dummy variables given bisim's assumption on deep-mind-control suite APIs
        low = -1.0
        high = 1.0
        self.action_space = DotMap()
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = [2]
        self.observation_space = DotMap()
        # D, H, W
        # before stack
        dummy_dict = {
            "RGB-Frame": (3, self.rl_image_size, self.num_cameras * self.rl_image_size),
            "DVS-Frame": (2, self.rl_image_size, self.num_cameras * self.rl_image_size),
            "DVS-Voxel-Grid": (5, self.rl_image_size, self.num_cameras * self.rl_image_size),
            "DVS-Stream": (None, 4),
            "E2VID-Frame": (1, self.rl_image_size, self.num_cameras * self.rl_image_size),
            "LiDAR-PCD": (None, 4),
            "LiDAR-BEV": (1, self.rl_image_size, self.num_cameras * self.rl_image_size),
            "Depth-Frame": (1, self.rl_image_size, self.num_cameras * self.rl_image_size),
        }


        if self.perception_type.__contains__("+"):
            # 多模态
            modals = self.perception_type.split("+")

            self.observation_space.shape = []
            for one_modal in modals:
                self.observation_space.shape.append(
                    dummy_dict[one_modal]
                )
            self.observation_space.dtype = np.dtype(np.float32)

        else:
            self.observation_space.shape = dummy_dict[self.perception_type]
            self.observation_space.dtype = np.dtype(np.float32)


        self.reward_range = None
        self.metadata = None
        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0]).astype(np.float32)

    def _dist_from_center_lane(self, vehicle, info):
        # assume on highway
        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_s = vehicle_waypoint.s
        vehicle_velocity = vehicle.get_velocity()  # Vecor3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        assert road_id is not None
        lane_id = int(vehicle_waypoint_closest_to_road.lane_id)
        goal_lane_id = lane_id

        current_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=False)
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        if goal_waypoint is None:
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id,
                                                           vehicle_s + carla_waypoint_discretization)

        if goal_waypoint is None:
            # print("Episode fail: goal waypoint is off the road! (frame %d)" % self.time_step)
            done, dist, vel_s = True, 100., 0.
            info['reason_episode_ended'] = 'off_road'

        else:
            goal_location = goal_waypoint.transform.location
            goal_xy = np.array([goal_location.x, goal_location.y])
            dist = np.linalg.norm(vehicle_xy - goal_xy)

            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))

            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left. (frame %d)" % self.time_step)
                info['reason_episode_ended'] = 'no_waypoints'
                done, vel_s = True, 0.

            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)
                done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        if vehicle_velocity.z > 1. and self.time_step < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_velocity.z, self.time_step))
            info['reason_episode_ended'] = 'carla_bug'
            done = True
        if vehicle_location.z > 0.5 and self.time_step < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_location.z, self.time_step))
            info['reason_episode_ended'] = 'carla_bug'
            done = True

        return dist, vel_s, speed, done

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # print('Collision (intensity {})'.format(intensity))
        self._collision_intensities_during_last_time_step.append(intensity)

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly

        return actor_poly_dict

    def _control_all_walkers(self):

        walker_behavior_params = self.scenario_params[self.selected_scenario]["walker_behavior"]

        # if walker is dead 
        # for walker in self.walker_actors:
        #     if not walker.is_alive:
        #         walker.destroy()
        #         self.walker_actors.remove(walker)

        # all_veh_locs = [
        #     [one_actor.get_transform().location.x, one_actor.get_transform().location.y]
        #     for one_actor in self.vehicle_actors
        # ]
        # all_veh_locs = np.array(all_veh_locs, dtype=np.float32)

        for walker in self.walker_actors:
            if walker.is_alive:
                # get location and velocity of the walker
                loc_x, loc_y = walker.get_location().x, walker.get_location().y
                vel_x, vel_y = walker.get_velocity().x, walker.get_velocity().y
                walker_loc = np.array([loc_x, loc_y], dtype=np.float32)

                # judge whether walker can cross the road
                # dis_gaps = np.linalg.norm(all_veh_locs - walker_loc, axis=1)
                # cross_flag = (dis_gaps >= walker_behavior_params["secure_dis"]).all()
                cross_prob = walker_behavior_params["cross_prob"]

                if loc_y > walker_behavior_params["border"]["y"][1]:
                    if self.time_step % self.max_fps == 0 and random.random() < cross_prob:
                        walker.apply_control(self.left)
                    # else:
                    #     if loc_x > walker_behavior_params["border"]["x"][1]:
                    #         walker.apply_control(self.backward)
                    #
                    #     elif loc_x > walker_behavior_params["border"]["x"][0]:
                    #         if vel_x > 0:
                    #             walker.apply_control(self.forward)
                    #         else:
                    #             walker.apply_control(self.backward)
                    #
                    #     else:
                    #         walker.apply_control(self.forward)

                elif loc_y > walker_behavior_params["border"]["y"][0]:
                    if vel_y > 0:
                        walker.apply_control(self.right)
                    else:
                        walker.apply_control(self.left)

                else:
                    if self.time_step % self.max_fps == 0 and random.random() < cross_prob:
                        walker.apply_control(self.right)
                    #
                    # else:
                    #     if loc_x > walker_behavior_params["border"]["x"][1]:
                    #         walker.apply_control(self.backward)
                    #
                    #     elif loc_x > walker_behavior_params["border"]["x"][0]:
                    #         if vel_x > 0:
                    #             walker.apply_control(self.forward)
                    #         else:
                    #             walker.apply_control(self.backward)
                    #
                    #     else:
                    #         walker.apply_control(self.forward)

    def _clear_all_actors(self):
        # remove all vehicles, walkers, and sensors (in case they survived)
        # self.world.tick()

        if 'vehicle' in dir(self) and self.vehicle is not None:
            for one_sensor_actor in self.sensor_actors:
                if one_sensor_actor.is_alive:
                    one_sensor_actor.stop()
                    one_sensor_actor.destroy()

        # # self.vidar_data['voltage'] = np.zeros((self.obs_size, self.obs_size), dtype=np.uint16)
        for actor_filter in ['vehicle.*', 'walker.*']:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    actor.destroy()

        # for one_vehicle_actor in self.vehicle_actors:
        #     if one_vehicle_actor.is_alive:
        #         one_vehicle_actor.destroy()

        # for one_walker_ai_actor in self.walker_ai_actors:
        #     if one_walker_ai_actor.is_alive:
        #         one_walker_ai_actor.stop()
        #         one_walker_ai_actor.destroy()

        # for one_walker_actor in self.walker_actors:
        #     if one_walker_actor.is_alive:
        #         one_walker_actor.destroy()


        # for actor_filter in ['vehicle.*', 'controller.ai.walker', 'walker.*', 'sensor*']:
        #     for actor in self.world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id == 'controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()

        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_actors = []
        self.walker_ai_actors = []

        # self.world.tick()
        # self.client.reload_world(reset_settings=True)

    def _set_seed(self, seed):
        if seed:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)  # Current CPU
            torch.cuda.manual_seed(seed)  # Current GPU
            torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

            self.tm.set_random_device_seed(seed)

    def reset(self, selected_scenario=None, selected_weather=None, seed=None):

        self._clear_all_actors()

        # self.client.reload_world(reset_settings = True)

        if selected_scenario is not None:
            self.reset_num = 0
            self.selected_scenario = selected_scenario

            # print("map:", self.scenario_params[self.selected_scenario]["map"])
            self.client.reload_world(reset_settings=True)
            self.world = self.client.load_world(self.scenario_params[self.selected_scenario]["map"])
            # print("reload done")

        if selected_weather is not None:
            self.selected_weather = selected_weather

        if self.reset_num == 0:

            self._set_dummy_variables()

            # self.world = self.client.load_world(
            #     map_name = self.scenario_params[self.selected_scenario]["map"],
            #     reset_settings = False
            # )
            # remove dynamic objects to prevent 'tables' and 'chairs' flying in the sky
            env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Dynamic)
            toggle1 = set([one_env_obj.id for one_env_obj in env_objs])
            env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Poles)  # street lights
            toggle2 = set([one_env_obj.id for one_env_obj in env_objs])
            env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Vegetation)  # 植被
            toggle3 = set([one_env_obj.id for one_env_obj in env_objs])

            objects_to_toggle = toggle1 | toggle2 | toggle3
            self.world.enable_environment_objects(objects_to_toggle, False)
            self.map = self.world.get_map()

            # bp
            self._init_blueprints()

            # spectator
            if self.is_spectator:
                self.spectator = self.world.get_spectator()
            else:
                self.spectator = None

            # tm
            self.tm = self.client.get_trafficmanager(self.carla_tm_port)
            self.tm_port = self.tm.get_port()
            self.tm.set_global_distance_to_leading_vehicle(2.0)
            #
            self._set_seed(seed)
            # lm
            self.lm = self.world.get_lightmanager()
            # self.lm.turn_off(self.lm.get_all_lights())



        # reset
        self.reset_sync_mode(False)
        # self.reset_sync_mode(True)

        self.reset_surrounding_vehicles()
        self.reset_special_vehicles()
        self.reset_walkers()
        self.reset_ego_vehicle()
        self.reset_weather()
        self.reset_sensors()

        self.reset_sync_mode(True)

        # spectator
        if self.spectator is not None:
            # First-perception
            self.spectator.set_transform(
                carla.Transform(self.vehicle.get_transform().location,
                                carla.Rotation(pitch=-float(10), yaw=-float(self.fov)))
            )
            # BEV
            # self.spectator.set_transform(
            #     carla.Transform(self.vehicle.get_transform().location + carla.Location(z=40),
            #                     carla.Rotation(pitch=-90)))

        self.time_step = 0
        self.dist_s = 0
        self.return_ = 0
        self.velocities = []

        self.reward = [0]
        self.perception_data = []
        self.last_action = None

        # MUST warm up !!!!!!
        # take some steps (here we set step=5) to get ready for the DVS camera, walkers, and vehicles
        # the warm-up step is set empirically
        obs = None
        # warm_up_max_steps = self.control_hz     # 15
        warm_up_max_steps = 5
        while warm_up_max_steps > 0:
            warm_up_max_steps -= 1
            obs, _, _, _ = self.step(None)

            # self.world.tick()


            # print("len:self.perception_data:", len(self.perception_data))
            
        # self.vehicle.set_autopilot(True, self.carla_tm_port)
        # while abs(self.vehicle.get_velocity().x) < 0.02:
        #     #             print("!!!take one init step", warm_up_max_steps, self.vehicle.get_control(), self.vehicle.get_velocity())
        #     self.world.tick()
        #     #             action = self.compute_steer_action()
        #     #             obs, _, _, _ = self.step(action=action)
        #     #             self.time_step -= 1
        #     warm_up_max_steps -= 1
        #     if warm_up_max_steps < 0 and self.dvs_data['events'] is not None:
        #         break
        # self.vehicle.set_autopilot(False, self.carla_tm_port)

        self.time_step = 0
        self.init_frame = self.frame
        self.reset_num += 1
        # print("carla env reset done.")

        return obs

    def reset_sync_mode(self, synchronous_mode=True):

        self.delta_seconds = 1.0 / self.max_fps
        # max_substep_delta_time = 0.005

        #         self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=synchronous_mode,
            fixed_delta_seconds=self.delta_seconds,
            # substepping=True, # 'SUBSTEP' mode is not necessary for NONE-DYNAMICS simulations
            # max_substep_delta_time=0.005,
            # max_substeps=int(self.delta_seconds/max_substep_delta_time)
            ))
        self.tm.set_synchronous_mode(synchronous_mode)

    def reset_surrounding_vehicles(self):
        total_surrounding_veh_num = 0
        veh_bp = self.bp_lib.filter('vehicle.*')
        veh_bp = [x for x in veh_bp if int(x.get_attribute('number_of_wheels')) == 4]

        for one_type in ["same_dir_veh", "oppo_dir_veh"]:
            # for one_type in ["same_dir_veh"]:

            one_type_params = self.scenario_params[self.selected_scenario][one_type]

            # print("now at:", one_type)

            for one_part in range(len(one_type_params)):

                veh_num = one_type_params[one_part]["num"]

                while veh_num > 0:
                    if one_type_params[one_part]["type"] != 0:
                        veh_bp = random.choice(one_type_params[one_part]["type"])
                        rand_veh_bp = self.bp_lib.find(veh_bp)
                    else:
                        rand_veh_bp = random.choice(veh_bp)

                    spawn_road_id = one_type_params[one_part]["road_id"]
                    spawn_lane_id = random.choice(
                        one_type_params[one_part]["lane_id"])
                    spawn_start_s = np.random.uniform(
                        one_type_params[one_part]["start_pos"][0],
                        one_type_params[one_part]["start_pos"][1],
                    )

                    veh_pos = self.map.get_waypoint_xodr(
                        road_id=spawn_road_id,
                        lane_id=spawn_lane_id,
                        s=spawn_start_s,
                    ).transform
                    veh_pos.location.z += 0.1

                    if rand_veh_bp.has_attribute('color'):
                        color = random.choice(rand_veh_bp.get_attribute('color').recommended_values)
                        rand_veh_bp.set_attribute('color', color)
                    if rand_veh_bp.has_attribute('driver_id'):
                        driver_id = random.choice(rand_veh_bp.get_attribute('driver_id').recommended_values)
                        rand_veh_bp.set_attribute('driver_id', driver_id)
                    rand_veh_bp.set_attribute('role_name', 'autopilot')
                    vehicle = self.world.try_spawn_actor(rand_veh_bp, veh_pos)

                    if vehicle is not None:
                        vehicle.set_autopilot(True, self.tm_port)
                        if np.random.uniform(0, 1) <= one_type_params[one_part]["beam_ratio"]:
                            vehicle.set_light_state(
                                # carla.VehicleLightState.HighBeam
                                carla.VehicleLightState.All
                            )

                        self.tm.auto_lane_change(vehicle, True)
                        self.tm.vehicle_percentage_speed_difference(
                            vehicle, np.random.uniform(one_type_params[one_part]["speed"][1],
                                                       one_type_params[one_part]["speed"][0]))
                        self.tm.ignore_lights_percentage(vehicle, 100)
                        self.tm.ignore_signs_percentage(vehicle, 100)
                        self.world.tick()

                        self.vehicle_actors.append(vehicle)

                        veh_num -= 1
                        total_surrounding_veh_num += 1
                        # print(f"\t spawn vehicle: {total_surrounding_veh_num}, at {veh_pos.location}")


    def reset_special_vehicles(self):
        special_veh_params = self.scenario_params[self.selected_scenario]["special_veh"]
        veh_bp = self.bp_lib.filter('vehicle.*')
        veh_bp = [x for x in veh_bp if int(x.get_attribute('number_of_wheels')) == 4]

        self.special_veh_lane_ids = []
        for one_part in range(len(special_veh_params)):
            veh_num = special_veh_params[one_part]["num"]

            while veh_num > 0:

                if special_veh_params[one_part]["type"] != 0:
                    # print("@@@:", special_veh_params[one_part]["type"])
                    rand_veh_bp = self.bp_lib.find(special_veh_params[one_part]["type"])
                else:
                    rand_veh_bp = random.choice(veh_bp)

                spawn_road_id = special_veh_params[one_part]["road_id"]
                spawn_lane_id = random.choice(
                    special_veh_params[one_part]["lane_id"])
                spawn_start_s = np.random.uniform(
                    special_veh_params[one_part]["start_pos"][0],
                    special_veh_params[one_part]["start_pos"][1],
                )

                veh_pos = self.map.get_waypoint_xodr(
                    road_id=spawn_road_id,
                    lane_id=spawn_lane_id,
                    s=spawn_start_s,
                ).transform
                veh_pos.location.z += 5
                veh_pos.rotation.pitch = special_veh_params[one_part]["pitch_range"] \
                    if special_veh_params[one_part]["pitch_range"] == 0 \
                    else np.random.uniform(special_veh_params[one_part]["pitch_range"][0],
                                           special_veh_params[one_part]["pitch_range"][1])
                veh_pos.rotation.yaw = special_veh_params[one_part]["yaw_range"] \
                    if special_veh_params[one_part]["yaw_range"] == 0 \
                    else np.random.uniform(special_veh_params[one_part]["yaw_range"][0],
                                           special_veh_params[one_part]["yaw_range"][1])
                veh_pos.rotation.roll = special_veh_params[one_part]["roll_range"] \
                    if special_veh_params[one_part]["roll_range"] == 0 \
                    else np.random.uniform(special_veh_params[one_part]["roll_range"][0],
                                           special_veh_params[one_part]["roll_range"][1])


                if rand_veh_bp.has_attribute('color'):
                    # print("color:", rand_veh_bp.get_attribute('color').recommended_values)
                    if special_veh_params[one_part]["color"] in rand_veh_bp.get_attribute('color').recommended_values:
                        rand_veh_bp.set_attribute('color', special_veh_params[one_part]["color"])
                    else:
                        color = random.choice(rand_veh_bp.get_attribute('color').recommended_values)
                        rand_veh_bp.set_attribute('color', color)

                if rand_veh_bp.has_attribute('driver_id'):
                    driver_id = random.choice(rand_veh_bp.get_attribute('driver_id').recommended_values)
                    rand_veh_bp.set_attribute('driver_id', driver_id)
                rand_veh_bp.set_attribute('role_name', 'autopilot')
                vehicle = self.world.try_spawn_actor(rand_veh_bp, veh_pos)

                if vehicle is not None:
                    self.special_veh_lane_ids.append(spawn_lane_id)

                    vehicle.open_door(carla.VehicleDoor.All)
                    vehicle.set_autopilot(False, self.tm_port)
                    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                    if np.random.uniform(0, 1) <= special_veh_params[one_part]["beam_ratio"]:
                        vehicle.set_light_state(
                            carla.VehicleLightState.HighBeam
                        )
                    self.world.tick()
                    self.vehicle_actors.append(vehicle)

                    veh_num -= 1



    def reset_walkers(self):
        walker_bp = self.bp_lib.filter('walker.*')
        total_surrounding_walker_num = 0

        walker_params = self.scenario_params[self.selected_scenario]["walker"]

        if len(walker_params) == 0:
            return

        walker_behavior_params = self.scenario_params[self.selected_scenario]["walker_behavior"]

        self.left = carla.WalkerControl(
            direction=carla.Vector3D(y=-1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))
        self.right = carla.WalkerControl(
            direction=carla.Vector3D(y=1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))

        self.forward = carla.WalkerControl(
            direction=carla.Vector3D(x=1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))
        self.backward = carla.WalkerControl(
            direction=carla.Vector3D(x=-1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))

        for one_part in range(len(walker_params)):

            walker_num = walker_params[one_part]["num"]

            while walker_num > 0:
                random.seed(1231)
                rand_walker_bp = random.choice(walker_bp)

                spawn_road_id = walker_params[one_part]["road_id"]
                spawn_lane_id = random.choice(
                    walker_params[one_part]["lane_id"])
                spawn_start_s = np.random.uniform(
                    walker_params[one_part]["start_pos"][0],
                    walker_params[one_part]["start_pos"][1],
                )

                walker_pos = self.map.get_waypoint_xodr(
                    road_id=spawn_road_id,
                    lane_id=spawn_lane_id,
                    s=spawn_start_s,
                ).transform
                walker_pos.location.z += 0.1

                # set as not invencible
                if rand_walker_bp.has_attribute('is_invincible'):
                    rand_walker_bp.set_attribute('is_invincible', 'false')

                # if rand_walker_bp.has_attribute('color'):
                #     color = random.choice(rand_walker_bp.get_attribute('color').recommended_values)
                #     rand_walker_bp.set_attribute('color', color)

                walker_actor = self.world.try_spawn_actor(rand_walker_bp, walker_pos)

                if walker_actor:
                    # print(walker_actor.attributes)
                    # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    # walker_controller_actor = self.world.spawn_actor(
                    #     walker_controller_bp, carla.Transform(), walker_actor)
                    # # start walker
                    # walker_controller_actor.start()
                    # # set walk to random point
                    # #             walker_controller_actor.go_to_location(world.get_random_location_from_navigation())
                    # rand_destination = carla.Location(
                    #     x=np.random.uniform(walker_params[one_part]["dest"]["x"][0], walker_params[one_part]["dest"]["x"][1]),
                    #     y=random.choice([walker_params[one_part]["dest"]["y"][0], walker_params[one_part]["dest"]["y"][1]]),
                    #     z=0.
                    # )
                    # walker_controller_actor.go_to_location(rand_destination)
                    # # random max speed (default is 1.4 m/s)
                    # walker_controller_actor.set_max_speed(
                    #     np.random.uniform(
                    #         walker_params[one_part]["speed"][0],
                    #         walker_params[one_part]["speed"][1]
                    #     ))
                    # self.walker_ai_actors.append(walker_controller_actor)

                    self.walker_actors.append(walker_actor)

                    self.world.tick()
                    walker_num -= 1
                    total_surrounding_walker_num += 1
                    # print(f"\t spawn walker: {total_surrounding_walker_num}, at {walker_pos.location}")

    def reset_ego_vehicle(self):

        self.vehicle = None

        # create vehicle
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        #         print(f"\tlen of self.vehicle_polygons: {len(self.vehicle_polygons[-1].keys())}")
        #         print(self.vehicle_polygons[-1].keys())
        ego_veh_params = self.scenario_params[self.selected_scenario]["ego_veh"]

        ego_spawn_times = 0
        max_ego_spawn_times = 10

        while True:
            # print("ego_spawn_times:", ego_spawn_times)d

            if ego_spawn_times > max_ego_spawn_times:

                ego_spawn_times = 0

                # print("\tspawn ego vehicle times > max_ego_spawn_times")
                self._clear_all_actors()
                self.reset_surrounding_vehicles()
                self.reset_special_vehicles()
                self.reset_walkers()

                self.vehicle_polygons = []
                vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
                self.vehicle_polygons.append(vehicle_poly_dict)

                continue

            # Check if ego position overlaps with surrounding vehicles
            overlap = False

            spawn_road_id = ego_veh_params["road_id"]
            if len(self.special_veh_lane_ids) == 0:
                spawn_lane_id = random.choice(ego_veh_params["lane_id"])
            else:
                spawn_lane_id = random.choice(self.special_veh_lane_ids)

            spawn_start_s = np.random.uniform(
                ego_veh_params["start_pos"][0],
                ego_veh_params["start_pos"][1],
            )

            veh_start_pose = self.map.get_waypoint_xodr(
                road_id=spawn_road_id,
                lane_id=spawn_lane_id,
                s=spawn_start_s,
            ).transform
            veh_start_pose.location.z += 0.1


            for idx, poly in self.vehicle_polygons[-1].items():
                poly_center = np.mean(poly, axis=0)
                ego_center = np.array([veh_start_pose.location.x, veh_start_pose.location.y])
                dis = np.linalg.norm(poly_center - ego_center)
                if dis > 8:
                    continue
                else:
                    overlap = True

                    break

            if not overlap:
                self.vehicle = self.world.try_spawn_actor(
                    self.bp_lib.find(ego_veh_params["type"]),
                    veh_start_pose
                )

            if self.vehicle is not None:

                self.vehicle_actors.append(self.vehicle)
                if self.selected_weather == "dense_fog":
                    self.vehicle.set_light_state(carla.VehicleLightState.Fog)
                    self.vehicle.set_light_state(carla.VehicleLightState.LowBeam)
                elif self.selected_weather == "midnight":
                    # self.vehicle.set_light_state(carla.VehicleLightState.Fog)
                    self.vehicle.set_light_state(carla.VehicleLightState.LowBeam)

                # AUTO pilot
                if self.ego_auto_pilot:
                    self.vehicle.set_autopilot(True, self.tm_port)

                    self.tm.distance_to_leading_vehicle(self.vehicle, 1)
                    self.tm.auto_lane_change(self.vehicle, True)
                    self.tm.vehicle_percentage_speed_difference(
                        self.vehicle, ego_veh_params["speed"])
                    self.tm.ignore_lights_percentage(self.vehicle, 100)
                    self.tm.ignore_signs_percentage(self.vehicle, 100)
                    # self.tm.force_lane_change(self.vehicle, True)
                else:
                    # immediate running
                    """
                    the driver will spend starting the car engine or changing a new gear. 
                    https://github.com/carla-simulator/carla/issues/3256
                    https://github.com/carla-simulator/carla/issues/1640
                    """
                    # self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
                    # self.world.tick()
                    # self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=False))

                    physics_control = self.vehicle.get_physics_control()
                    physics_control.gear_switch_time = 0.01
                    physics_control.damping_rate_zero_throttle_clutch_engaged = physics_control.damping_rate_zero_throttle_clutch_disengaged
                    self.vehicle.apply_physics_control(physics_control)
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1, manual_gear_shift=True, gear=1))
                    # pass

                break

            else:
                ego_spawn_times += 1
                time.sleep(0.01)
                # print("ego_spawn_times:", ego_spawn_times)

        self.world.tick()

    def reset_sensors(self):

        # [one_sensor.stop() for one_sensor in self.sensors]

        # data
        if self.BEV:
            self.bev_data = {'frame': 0, 'timestamp': 0.0, 'img': np.zeros((2048, 2048, 3), dtype=np.uint8)}

        if self.TPV:
            self.video_data = {'frame': 0, 'timestamp': 0.0, 'img': np.zeros((1024, 1024, 3), dtype=np.uint8)}

        self.rgb_data = {'frame': 0, 'timestamp': 0.0,
                         'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)}

        self.depth_data = {'frame': 0, 'timestamp': 0.0,
                            'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 1), dtype=np.uint8)}

        self.dvs_data = {'frame': 0, 'timestamp': 0.0,
                         'events': None,
                         'latest_time': np.zeros(
                             (self.rl_image_size, self.rl_image_size * self.num_cameras), dtype=np.int64),
                         'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8),
                         'denoised_img': None,
                         # 'rec-img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 1), dtype=np.uint8),
                         }

        self.lidar_data = {'frame': 0, 'timestamp': 0.0,
                           'BEV': np.zeros((self.rl_image_size, self.rl_image_size, 1), dtype=np.uint8),
                           'PCD': None,}

        # self.dvs_type = dvs_type
        # self.dvs_param = dvs_param
        # DVS_TYPE = ASYN_CONSTANT_EVENT_NUMBER
        # DVS_TYPE = ASYN_CONSTANT_TEMPORAL_WINDOW
        # DVS_TYPE = SYN_TEMPORAL_WINDOW
        self.dvs_cache = []


        self.frame = None

        #         def on_tick_func(data):
        #             self.dvs_data["events"] = None
        #         self.world.on_tick(on_tick_func)
        # Bird Eye View
        if self.BEV:
            def __get_bev_data__(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.bev_data['frame'] = data.frame
                self.bev_data['timestamp'] = data.timestamp
                self.bev_data['img'] = array

            self.bev_camera_rgb = self.world.spawn_actor(
                self.bev_camera_bp,
                carla.Transform(carla.Location(z=22), carla.Rotation(pitch=-90, yaw=90, roll=-90)),
                attach_to=self.vehicle)
            self.bev_camera_rgb.listen(lambda data: __get_bev_data__(data))
            self.sensor_actors.append(self.bev_camera_rgb)


        # Third Person View
        if self.TPV:
            def __get_video_data__(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.video_data['frame'] = data.frame
                self.video_data['timestamp'] = data.timestamp
                self.video_data['img'] = array

            self.video_camera_rgb = self.world.spawn_actor(
                self.video_camera_bp,
                carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.video_camera_rgb.listen(lambda data: __get_video_data__(data))
            self.sensor_actors.append(self.video_camera_rgb)

        #         print("\t video sensor init done.")

        # we'll use up to five cameras, which we'll stitch together
        location = carla.Location(x=1, z=1.5)

        # Perception RGB sensor
        def __get_rgb_data__(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_data['frame'] = data.frame
            self.rgb_data['timestamp'] = data.timestamp
            self.rgb_data['img'] = array

        self.rgb_camera = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle)
        self.rgb_camera.listen(lambda data: __get_rgb_data__(data))
        self.sensor_actors.append(self.rgb_camera)


        # Perception Depth sensor
        if self.perception_type.__contains__("Depth"):
            def __get_depth_data__(data):
                # data.convert(carla.ColorConverter.Depth)
                data.convert(carla.ColorConverter.LogarithmicDepth) # leading to better precision for small distances at the expense of losing it when further away.
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.depth_data['frame'] = data.frame
                self.depth_data['timestamp'] = data.timestamp
                # self.depth_data['img'] = array[:,:,0][..., np.newaxis]
                # self.depth_data['img'] = array[:,:,1][..., np.newaxis]
                # self.depth_data['img'] = array[:,:,2][..., np.newaxis]
                # self.depth_data['img'] = array
                self.depth_data['img'] = array[:, :, 0][..., np.newaxis]

            self.depth_camera = self.world.spawn_actor(
                self.depth_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
                attach_to=self.vehicle)
            self.depth_camera.listen(lambda data: __get_depth_data__(data))
            self.sensor_actors.append(self.depth_camera)
            


        # Perception DVS sensor
        if self.perception_type.__contains__("DVS"):
            def __get_dvs_data__(data):
                #             print("get_dvs_data:", one_camera_idx)
                events = np.frombuffer(data.raw_data, dtype=np.dtype([
                    ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)]))
                self.dvs_data['frame'] = data.frame
                self.dvs_data['timestamp'] = data.timestamp

                # img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
                # img[events[:]['y'], events[:]['x'], events[:]['pol'] * 2] = 255
                # self.dvs_data['img'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size, :] = img

                x = events['x'].astype(np.int32)
                y = events['y'].astype(np.int32)
                p = events['pol'].astype(np.float32)
                t = events['t'].astype(np.float32)
                events = np.column_stack((x, y, p, t))  # (event_num, 4)

                self.dvs_data['events'] = events
                self.dvs_data['events'] = self.dvs_data['events'][np.argsort(self.dvs_data['events'][:, -1])]
                self.dvs_data['events'] = self.dvs_data['events'].astype(np.float32)
                # init done.
                # print(self.dvs_data['events'][:, -1])      # event是按时间递增排的

                # DVS-frame
                img = np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)
                # print("unique:", np.unique(self.dvs_data['events'][:, 2]))
                img[self.dvs_data['events'][:, 1].astype(np.int),
                    self.dvs_data['events'][:, 0].astype(np.int),
                    self.dvs_data['events'][:, 2].astype(np.int) * 2] = 255
                self.dvs_data['img'] = img
                # if self.DENOISE:
                #     self.dvs_data['denoised_img'] = img.copy()


            self.dvs_camera = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
                attach_to=self.vehicle)
            self.dvs_camera.listen(lambda data: __get_dvs_data__(data))
            self.sensor_actors.append(self.dvs_camera)


        # Perception LiDAR sensor
        if self.perception_type.__contains__("LiDAR"):
            def __get_lidar_data__(data):
                self.lidar_data['frame'] = data.frame
                self.lidar_data['timestamp'] = data.timestamp

                lidar_range = 2.0 * float(self.lidar_range)  # range

                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                # print("@@@:", points.shape,)
                self.lidar_data['PCD'] = points

                pos = np.array(points[:, :2])       # 原本图像的坐标起点是图片的中心
                # print(pos.max(), pos.min())
                pos += (0.5 * lidar_range, 0.5 * lidar_range)    # 转变为左上角起点
                pos = np.fabs(pos)
                pos = pos.astype(np.int32)
                pos = np.reshape(pos, (-1, 2))
                lidar_img = np.zeros((int(lidar_range), int(lidar_range), 1), dtype=np.uint8)
                # lidar_img[tuple(pos.T)] = (255, 255, 255)
                lidar_img[tuple(pos.T)] = 255
                # resize
                lidar_img = cv2.resize(lidar_img, (self.rl_image_size, self.rl_image_size))
                lidar_img = lidar_img[..., np.newaxis]
                self.lidar_data['BEV'] = lidar_img


            self.lidar_camera = self.world.spawn_actor(
                self.lidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
                attach_to=self.vehicle)
            self.lidar_camera.listen(lambda data: __get_lidar_data__(data))
            self.sensor_actors.append(self.lidar_camera)


        # Collision Sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self._collision_intensities_during_last_time_step = []
        self.sensor_actors.append(self.collision_sensor)


        self.world.tick()


    def reset_weather(self):
        assert self.selected_weather in self.weather_params.keys()

        weather_params = self.weather_params[self.selected_weather]

        self.weather = self.world.get_weather()

        self.weather.cloudiness = weather_params["cloudiness"]
        self.weather.precipitation = weather_params["precipitation"]
        self.weather.precipitation_deposits = weather_params["precipitation_deposits"]
        self.weather.wind_intensity = weather_params["wind_intensity"]
        self.weather.fog_density = weather_params["fog_density"]
        self.weather.fog_distance = weather_params["fog_distance"]
        self.weather.wetness = weather_params["wetness"]
        self.weather.sun_azimuth_angle = weather_params["sun_azimuth_angle"]
        self.weather.sun_altitude_angle = weather_params["sun_altitude_angle"]

        self.world.set_weather(self.weather)


    def step(self, action):
        rewards = []
        next_obs, done, info = None, None, None

        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(action, self.delta_seconds)
            # next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break

        return next_obs, np.mean(rewards), done, info  # just last info?


    def _control_spectator(self):
        if self.spectator is not None:
            # First-perception
            self.spectator.set_transform(
                carla.Transform(self.vehicle.get_transform().location + carla.Location(z=2),
                                self.vehicle.get_transform().rotation)
            )
            # BEV
            # self.spectator.set_transform(
            #     carla.Transform(self.vehicle.get_transform().location + carla.Location(z=40),
            #                     carla.Rotation(pitch=-90)))


    def _simulator_step(self, action, dt=0.1):

        if action is None and self.last_action is not None:
            action = self.last_action

        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            self.last_action = action

        else:
            throttle, steer, brake = 0., 0., 0.

        assert 0.0 <= throttle <= 1.0
        assert -1.0 <= steer <= 1.0
        assert 0.0 <= brake <= 1.0
        vehicle_control = carla.VehicleControl(
            throttle=throttle,  # [0.0, 1.0]
            steer=steer,  # [-1.0, 1.0]
            brake=brake,  # [0.0, 1.0]
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(vehicle_control)

        self._control_spectator()
        self._control_all_walkers()

        # Advance the simulation and wait for the data.
        #         self.dvs_data["events"] = None
        self.frame = self.world.tick()


        # if self.spectator is not None:
        #     self.spectator.set_transform(
        #         carla.Transform(self.ego_vehicle.get_transform().location + carla.Location(z=40),
        #         carla.Rotation(pitch=-90)))

        info = {}
        info['reason_episode_ended'] = ''
        dist_from_center, vel_s, speed, done = self._dist_from_center_lane(self.vehicle, info)
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.
        colliding = float(collision_intensities_during_last_time_step > 0.)

        if colliding:
            self.collide_count += 1
        else:
            self.collide_count = 0

        if self.collide_count >= 20:
            # print("Episode fail: too many collisions ({})! (collide_count: {})".format(speed, self.collide_count))
            info['reason_episode_ended'] = 'collisions'
            done = True

        # reward = vel_s * dt / (1. + dist_from_center) - 1.0 * colliding - 0.1 * brake - 0.1 * abs(steer)
        # collision_cost = 0.0001 * collision_intensities_during_last_time_step

        # [Reward 1]
        # reward = vel_s * dt - collision_cost - abs(steer)

        # [Reward 2]
        # reward = vel_s * dt / (1. + dist_from_center) - 1.0 * colliding - 0.1 * brake - 0.1 * abs(steer)

        # [Reward 3]
        # reward = vel_s * dt / (1. + dist_from_center) - collision_cost - 0.1 * brake - 0.1 * abs(steer)

        # [Reward 4]
        collision_cost = 0.001 * collision_intensities_during_last_time_step
        reward = vel_s * dt - collision_cost - 0.1 * brake - 0.1 * abs(steer)

        self.reward.append(reward)
        # print("vel_s:", vel_s, "speed:", speed)

        self.dist_s += vel_s * self.delta_seconds
        self.return_ += reward

        self.time_step += 1

        next_obs = {
            'RGB-Frame': self.rgb_data['img'],
        }

        # next_obs.update({"DVS-Stream": self.dvs_data["events"]})


        if self.TPV:
            next_obs.update({'video-frame': self.video_data['img']})

        if self.BEV:
            next_obs.update({'BEV-frame': self.bev_data['img']})


        if self.perception_type.__contains__("+"):
            # 多模态
            modals = self.perception_type.split("+")

        else:
            modals = [self.perception_type]

        for one_modal in modals:
            if one_modal == "RGB-Frame":
                next_obs.update({one_modal: self.rgb_data['img']})


            elif one_modal == "DVS-Stream":
                if self.dvs_data['events'] is None:
                    next_obs.update({one_modal: np.zeros(shape=(1, 4), dtype=np.float32) * 1e-6})
                else:
                    next_obs.update({one_modal: self.dvs_data['events']})


            elif one_modal == "DVS-Frame":
                next_obs.update({one_modal: self.dvs_data['img']})
                # print("\t", "event_num:", len(self.dvs_data["events"]), self.dvs_data["events"][:5, -1])
                # if self.DENOISE:
                #     next_obs.update({'Denoised-DVS-frame': self.dvs_data['denoised_img']})

            elif one_modal == "DVS-Voxel-Grid":
                next_obs.update({"DVS-Frame": self.dvs_data['img']})

                # (5, 128, 128)
                num_bins, height, width = 5, self.rl_image_size, int(self.rl_image_size * self.num_cameras)
                voxel_grid = np.zeros(shape=(num_bins, height, width), dtype=np.float32).ravel()

                # get events
                events = self.dvs_data["events"]  # (x, y, p, t)

                if events is not None and len(events) > 0:
                    """events to pytorch.tensor"""

                    events_torch = torch.from_numpy(events).clone()
                    # voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=self.device).flatten()
                    voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
                    # normalize the event timestamps so that they lie between 0 and num_bins
                    last_stamp = events_torch[-1, -1]
                    first_stamp = events_torch[0, -1]
                    deltaT = last_stamp - first_stamp

                    if deltaT == 0: deltaT = 1.0

                    events_torch[:, -1] = (num_bins - 1) * (events_torch[:, -1] - first_stamp) / deltaT
                    ts = events_torch[:, -1]
                    xs = events_torch[:, 0].long()
                    ys = events_torch[:, 1].long()
                    pols = events_torch[:, 2].float()
                    pols[pols == 0] = -1  # polarity should be +1 / -1

                    tis = torch.floor(ts)
                    tis_long = tis.long()
                    dts = ts - tis
                    vals_left = pols * (1.0 - dts.float())
                    vals_right = pols * dts.float()

                    valid_indices = tis < num_bins
                    valid_indices &= tis >= 0
                    voxel_grid.index_add_(dim=0,
                                          index=xs[valid_indices] + ys[valid_indices]
                                                * width + tis_long[valid_indices] * width * height,
                                          source=vals_left[valid_indices])

                    valid_indices = (tis + 1) < num_bins
                    valid_indices &= tis >= 0

                    voxel_grid.index_add_(dim=0,
                                          index=xs[valid_indices] + ys[valid_indices] * width
                                                + (tis_long[valid_indices] + 1) * width * height,
                                          source=vals_right[valid_indices])

                    voxel_grid = voxel_grid.view(num_bins, height, width)
                    voxel_grid = voxel_grid.cpu().numpy()
                    # print("voxel_grid:", np.max(voxel_grid), np.min(voxel_grid))
                    next_obs.update({'DVS-Voxel-Grid': np.transpose(voxel_grid, (1, 2, 0))})

                else:
                    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
                    next_obs.update({"DVS-Voxel-Grid": np.transpose(voxel_grid, (1, 2, 0))})

            elif one_modal == "Depth-Frame":
                # print("self.depth_data['img']:", self.depth_data['img'].shape)
                next_obs.update({one_modal: self.depth_data['img']})

            elif one_modal == "LiDAR-BEV":
                next_obs.update({one_modal: self.lidar_data['BEV']})

            elif one_modal == "LiDAR-PCD":
                next_obs.update({one_modal: self.lidar_data['PCD']})


        info['crash_intensity'] = collision_intensities_during_last_time_step
        info['throttle'] = throttle
        info['steer'] = steer
        info['brake'] = brake
        info['distance'] = vel_s * dt

        if self.time_step >= self.max_episode_steps:
            info['reason_episode_ended'] = 'success'
            # print("Episode success: I've reached the episode horizon ({}).".format(self.max_episode_steps))
            done = True
        #         if speed < 0.02 and self.time_step >= 8 * (self.fps) and self.time_step % 8 * (self.fps) == 0:  # a hack, instead of a counter
        if speed < 0.02 and self.time_step >= self.min_stuck_steps and self.time_step % self.min_stuck_steps == 0:  # a hack, instead of a counter
            # print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(speed, self.time_step))
            info['reason_episode_ended'] = 'stuck'
            done = True

        return next_obs, reward, done, info


    def finish(self):
        print('destroying actors.')
        actor_list = self.world.get_actors()
        for one_actor in actor_list:
            one_actor.destroy()
        time.sleep(0.5)
        print('done.')

