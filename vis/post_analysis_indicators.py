import os.path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline



# domain = "highbeam"
# domain = "tunnel"
domain = "jaywalk"
# domain = "crash"
# domain = "normal"


weather = "midnight"
# weather = "hard_rain"
# weather = "cloudy"


seeds = [
    "111",
    "222",
    "333",
    # "444",
    # "555",
]


logfiles = [
    fr"E:\logs\final\carla+{domain}+{weather}+mlr+RGB-Frame+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+sac+RGB-Frame+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+mlr+DVS-Voxel-Grid+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+sac+DVS-Voxel-Grid+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+DVS-Frame+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+DVS-Voxel-Grid+pixelCarla098",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelInputFusion",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCat",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCatSep",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCrossFusion",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelEFNet",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelFPNNet",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelRENet",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV4",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV4_Repel",
    fr"E:\logs\final\carla+{domain}+{weather}+sac+RGB-Frame+DVS-Voxel-Grid+pixelConNewV4",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV2",
    # f"../logs/final/carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV2+111+cat.log",
    # f"../logs/final/carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV2+111+plus.log",
]


for one_logfile in logfiles:
    ER, DT = [], []
    file_exists = True
    print("one_logfile:", one_logfile)

    for one_seed in seeds:
        # print("\tSEED:", one_seed)
        if not os.path.exists(one_logfile + f"+{one_seed}.log"):
            file_exists = False
            break
        with open(one_logfile + f"+{one_seed}.log") as file:
            lines = file.readlines()
            for one_line_idx, one_line in enumerate(lines):
                if one_line.startswith("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@at step: [109999]"):
                    # print("111111")

                    for i in range(one_line_idx+1, len(lines)):
                        if lines[i].startswith("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"):
                            break
                        elif lines[i].startswith("distance_driven_each_episode:"):
                            DT += eval(lines[i].split(":")[1].strip())
                        elif lines[i].startswith("reward_each_episode:"):
                            ER += eval(lines[i].split(":")[1].strip())

                else:
                    continue

    if file_exists:
        print(f"===> DT: "
              f"{np.around(np.mean(DT), 3)}±"
              f"{np.around(np.std(DT), 3)}")
        print(f"===> ER: "
              f"{np.around(np.mean(ER), 3)}±"
              f"{np.around(np.std(ER), 3)}")
    else:
        print("\tnot exists")
    time.sleep(0.1)
