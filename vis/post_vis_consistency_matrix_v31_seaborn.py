# https://zhuanlan.zhihu.com/p/448932423
# https://github.com/SarthakJariwala/seaborn-image

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn_image as isns
import cv2



# domain = "highbeam"
# domain = "crash"
domain = "jaywalk"


weather = "midnight"


agent = "deepmdp"

perception = "RGB-Frame+DVS-Voxel-Grid"
# encoder = "pixelConNewV3"
encoder = "pixelConNewV4"
# seed = 111
# seed = 222
seed = 333
# step = 100000
step = 10000

dataset = torch.load(
    rf'E:\logs\final\carla+{domain}+{weather}+{agent}+{perception}+{encoder}+{seed}\embed\consistency_params_{step}.pt',
    # rf'E:\data\xuhr\nips2023\seperator\carla+{domain}+{weather}+{agent}+{perception}+{encoder}+{seed}\embed\consistency_params_{step}.pt',
    map_location=torch.device('cpu')
)


print(dataset.keys())

com_to_else_logits = dataset['com_to_else_logits']
# dvs_to_rgb_logits = dataset['dvs_to_rgb_logits']
# rgb_to_dvs_logits = dataset['rgb_to_dvs_logits']

print("rgb_h_query:", dataset['rgb_h_query'].shape)      # (32, 50)
print("com_h_query:", dataset['com_h_query'].shape)      # (32, 50)
print("dvs_h_query:", dataset['dvs_h_query'].shape)      # (32, 50)
print("com_to_else_logits:", com_to_else_logits.shape)      # (32, 65)
# print("\t", com_to_else_logits.min(), com_to_else_logits.max())
# print("dvs_to_rgb_logits:", dvs_to_rgb_logits.shape)        # (32, 33)
# print("\t", dvs_to_rgb_logits.min(), dvs_to_rgb_logits.max())
# print("rgb_to_dvs_logits:", rgb_to_dvs_logits.shape)        # (32, 33)
# print("\t", rgb_to_dvs_logits.min(), rgb_to_dvs_logits.max())

print("dvs <-> rgb")
# print("\t", (rgb_to_dvs_logits+dvs_to_rgb_logits).min(), (rgb_to_dvs_logits+dvs_to_rgb_logits).max())

# , cbar_ticks=[-1.0, -0.5, 0, 0.5, 1.0]
ax1 = isns.imgplot(com_to_else_logits, cmap="deep", orientation="h", cbar_ticks=[-0.5, 0, 0.5, 1.0], robust=True, perc=(0,99.99))
# ax2 = isns.imgplot(rgb_to_dvs_logits+dvs_to_rgb_logits, cmap="deep", orientation="h", cbar_ticks=[-0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0], robust=True, perc=(0,99.99))


plt.savefig("./cm.pdf")
plt.show()
