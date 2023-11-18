import os
import imageio
import numpy as np
from PIL import ImageFont, ImageDraw, Image

class VideoRecorder(object):
    def __init__(self, dir_name, min_fps, max_fps):
        self.dir_name = dir_name
        self.min_fps = min_fps
        self.max_fps = max_fps

        self.bev_frames = []
        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.denoised_dvs_frames = []
        self.dvs_rec_frames = []
        self.depth_frames = []
        self.lidar_bev_frames = []
        self.vidar_frames = []

        self.enabled = False

    def init(self, enabled=True):

        self.bev_frames = []
        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.denoised_dvs_frames = []
        self.dvs_rec_frames = []
        self.depth_frames = []
        self.lidar_bev_frames = []
        self.vidar_frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs, env, vehicle=None):

        if self.enabled:
            if "video-frame" in obs.keys():

                video_frame = obs["video-frame"].copy()

                if vehicle is not None:
                    height, width = video_frame.shape[0:2]  # 600, 800

                    video_frame = Image.fromarray(video_frame)  # .convert("P")
                    draw = ImageDraw.Draw(video_frame)
                    # print(video_frame.mode)

                    control = vehicle.get_control()
                    velocity = vehicle.get_velocity()

                    dw = width - 150
                    dh = 20

                    # rectangle
                    draw.rectangle(xy=((dw-10, dh-10), (dw+115, dh+160)),
                                   outline=(147,137,148), fill=(92,91,107),
                                   width=3)

                    # annotation
                    draw.text((dw, dh), f"throttle: {control.throttle:.5f}", fill=(255, 255, 255))
                    draw.text((dw, dh + 20), f"steer: {control.steer:.5f}", fill=(255, 255, 255))
                    draw.text((dw, dh + 40), f"brake: {control.brake:.5f}", fill=(255, 255, 255))
                    draw.text((dw, dh + 60), f"vx: {velocity.x:.5f}", fill=(255, 255, 255))
                    draw.text((dw, dh + 80), f"vy: {velocity.y:.5f}", fill=(255, 255, 255))

                    draw.text((dw, dh + 120), f"MAX FPS: {self.max_fps}", fill=(255, 255, 255))
                    draw.text((dw, dh + 140), f"MIN FPS: {self.min_fps}", fill=(255, 255, 255))


                    # video_frame.show()
                    video_frame = np.array(video_frame)

                self.video_frames.append(video_frame)

            if "BEV-Frame" in obs.keys():
                self.bev_frames.append(obs["BEV-Frame"].copy())
            if "RGB-Frame" in obs.keys():
                self.rgb_frames.append(obs["RGB-Frame"].copy())
            if "DVS-Frame" in obs.keys():
                self.dvs_frames.append(obs["DVS-Frame"].copy())
            # if "Denoised-DVS-frame" in obs.keys():
            #     self.denoised_dvs_frames.append(obs["Denoised-DVS-frame"].copy())
            if "Depth-Frame" in obs.keys():
                self.depth_frames.append(obs["Depth-Frame"].copy())
            if "E2VID-Frame" in obs.keys():
                self.dvs_rec_frames.append(obs["E2VID-Frame"].copy())
            if "LiDAR-BEV" in obs.keys():
                self.lidar_bev_frames.append(obs["LiDAR-BEV"].copy())
            # if "Vidar-Frame" in obs.keys():
            #     self.vidar_frames.append(obs["Vidar-Frame"].copy())



    def save(self, file_name, type="mp4"):
        if self.enabled:
            bev_frames_path = os.path.join(self.dir_name, file_name + f"-bev.{type}")
            video_frames_path = os.path.join(self.dir_name, file_name + f"-video.{type}")
            rgb_frames_path = os.path.join(self.dir_name, file_name + f"-rgb.{type}")
            dvs_frames_path = os.path.join(self.dir_name, file_name + f"-dvs.{type}")
            # denoised_dvs_frames_path = os.path.join(self.dir_name, file_name + f"-denoised-dvs.{type}")
            depth_frames_path = os.path.join(self.dir_name, file_name + f"-depth.{type}")
            dvs_rec_frames_path = os.path.join(self.dir_name, file_name + f"-rec-dvs.{type}")
            lidar_bev_frames_path = os.path.join(self.dir_name, file_name + f"-lidar-bev.{type}")
            # vidar_frames_path = os.path.join(self.dir_name, file_name + f"-vidar.{type}")
            perception_frames_path = os.path.join(self.dir_name, file_name + f"-perception.{type}")

            if len(self.bev_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(bev_frames_path, self.bev_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(bev_frames_path, self.bev_frames, duration=1 / self.max_fps)


            if len(self.video_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(video_frames_path, self.video_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(video_frames_path, self.video_frames, duration=1 / self.max_fps)

            if len(self.rgb_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(rgb_frames_path, self.rgb_frames, fps=self.min_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(rgb_frames_path, self.rgb_frames, duration=1 / self.max_fps)

            if len(self.dvs_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(dvs_frames_path, self.dvs_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(dvs_frames_path, self.dvs_frames, duration=1 / self.max_fps)

            # if len(self.denoised_dvs_frames) > 0:
            #     if type == "mp4":
            #         imageio.mimsave(denoised_dvs_frames_path, self.denoised_dvs_frames, fps=self.max_fps, macro_block_size=2)
            #     elif type == "gif":
            #         imageio.mimsave(denoised_dvs_frames_path, self.denoised_dvs_frames, duration=1 / self.max_fps)

            if len(self.depth_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(depth_frames_path, self.depth_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(depth_frames_path, self.depth_frames, duration=1 / self.max_fps)

            if len(self.dvs_rec_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(dvs_rec_frames_path, self.dvs_rec_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(dvs_rec_frames_path, self.dvs_rec_frames, duration=1 / self.max_fps)

            if len(self.lidar_bev_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(lidar_bev_frames_path, self.lidar_bev_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(lidar_bev_frames_path, self.lidar_bev_frames, duration=1 / self.max_fps)

            # if len(self.vidar_frames) > 0:
            #     if type == "mp4":
            #         imageio.mimsave(vidar_frames_path, self.vidar_frames, fps=self.max_fps, macro_block_size=2)
            #     elif type == "gif":
            #         imageio.mimsave(vidar_frames_path, self.vidar_frames, duration=1 / self.max_fps)
