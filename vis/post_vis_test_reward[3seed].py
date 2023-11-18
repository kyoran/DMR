import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns

"""
https://www.guyuehome.com/37664
https://github.com/feidieufo/homework/blob/master/util/plt_example.py
"""

plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size'] = 16



# domain = "highbeam"
# domain = "tunnel"
domain = "jaywalk"
# domain = "crash"
# domain = "normal"


weather = "midnight"
# weather = "hard_rain"
# weather = "cloudy"
weight = 0.2
# remain_last = True
remain_last = False

vis_what = "train"
# vis_what = "test"
smooth_alg = "tensorb"
# smooth_alg = "nearest"


seeds = [
    "111",
    "222",
    "333",
]


xnew = np.linspace(0, 110000, num=110)


logfiles = [
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+pixelCarla098",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+DVS-Frame+pixelCarla098",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+DVS-Voxel-Grid+pixelCarla098",

    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelInputFusion",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCat",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCatSep",

    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCrossFusion",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelEFNet",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelFPNNet",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelRENet",

    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV2",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV3",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV4",

    # f"../logs/final/carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV2+111+cat.log",
    # f"../logs/final/carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV2+111+plus.log",


    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelInputFusion",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelCat",
    # fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV4_Repel",
    fr"E:\logs\final\carla+{domain}+{weather}+deepmdp+RGB-Frame+DVS-Voxel-Grid+pixelConNewV4",

]
labels=[
	# "RGB-Frame",
	# "DVS-Frame",
    # "DVS-Voxel-Grid",
    # "InputFusion",
    # "LateFusion[Mono]",
    # "LateFusion[Tri]",
    "TransFuser",
    "EFNet",
    "FPNet",
    "RENet",
    "Ours",

    # "1 Branch",
    # "2 Branch",
    # "+ Repel",
    # "+ Rec (ours)",

]

# colors = [
#     '#8dd3c7',
#     '#bebada',
#     '#fb8072',
#     '#80b1d3',
#     '#fdb462',
#     '#b3de69',
#     '#fccde5',
#     '#d9d9d9',
#     '#bc80bd',
#     '#ccebc5',
#     '#ffed6f',
# ]


colors = [
    # 'b', 'r', 'g', 'y', 'm'
    '#7a68a5',
    '#a60628',
    '#348abd',
    '#d55e00',
    '#467821',
    '#c283a6',
    'm',
    '#000000',  # 全黑
]


min_step_xlim, max_step_lim, step_span = 0, 10, 6
min_step_xlim, max_step_lim, step_span = 0, 9, 4
# min_step_xlim, max_step_lim, step_span = 0, 8, 5
"""
b---blue c---cyan g---green k----black
m---magenta r---red w---white y----yellow
"""


fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

plt.title(f"{domain}-{weather}")


def _tensorboard_smoothing(values, smooth = 0.98, remain_last=False):
    """不需要传入step"""
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = smooth + 1
    x = values[0]
    res = [x]
    for i in range(1, len(values)):
        x = x * smooth + values[i]  # 指数衰减
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    if remain_last:
        res[-1] = values[-1]
    return res


def _nearest_smoothing(scalars, weight=0.98, remain_last=False):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    if remain_last:
        smoothed[-1] = scalars[-1]
    return smoothed


for iii, one_logfile in enumerate(logfiles):
    print("now at:", iii, one_logfile)

    new_train_steps, new_train_rewards = [], []     # 用于sns.tsplot画图
    new_test_steps, new_test_rewards = [], []     # 用于sns.tsplot画图

    for one_seed in seeds:
        # if "NewV4" in one_logfile:
        #     one_seed = int(one_seed) + 222
        print("\tSEED:", one_seed)
        train_steps, train_rewards = [], []
        test_steps, test_rewards = [], []

        # print(test_rewards)

        with open(one_logfile+f"+{one_seed}\\eval.log") as file:
            lines = file.readlines()
            for one_line in lines:
                train_dict = eval(one_line[:-1])

                train_steps.append(train_dict["step"])
                train_rewards.append(train_dict[f"{domain}_{weather}_episode_reward"])


            # train
            train_ynew10 = np.interp(xnew, train_steps, train_rewards)
            if smooth_alg == "tensorb":
                train_ynew10 = _tensorboard_smoothing(train_ynew10, weight, remain_last)
            if smooth_alg == "nearest":
                train_ynew10 = _nearest_smoothing(train_ynew10, weight, remain_last)     #

            # test
            # test_ynew10 = np.interp(xnew, test_steps, test_rewards)
            # if smooth_alg == "tensorb":
            #     test_ynew10 = _tensorboard_smoothing(test_ynew10, weight, remain_last)
            # if smooth_alg == "nearest":
            #     test_ynew10 = _nearest_smoothing(test_ynew10, weight, remain_last)

            new_train_steps.append(xnew)
            new_train_rewards.append(train_ynew10)

            # new_test_steps.append(xnew)
            new_test_steps.append(test_steps)
            # new_test_rewards.append(test_ynew10)
            # print("test_rewards:", len(test_rewards))
            new_test_rewards.append(test_rewards)


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Train ↓
    if vis_what == "train":
        new_train_steps = np.array(new_train_steps)       # (2, 100)
        new_train_rewards = np.array(new_train_rewards)   # (2, 100)
        # print("\tnew_train_rewards:", new_train_rewards.shape)
        g = sns.tsplot(time=xnew/1e4, data=new_train_rewards, color=colors[iii], condition=labels[iii])

        # ax[0].plot(np.array(train_steps)/1e4, train_rewards, color=colors[iii], alpha=0.1)
        # ax[0].plot(np.array(train_steps)/1e4, smooth(train_rewards, 0.98), color=colors[iii], label=labels[iii])
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Train ↑


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ TEST ↓

    # test_steps = np.array(test_steps)
    # sorted_idx = np.argsort(test_steps)
    #
    # test_steps = test_steps[sorted_idx]
    # test_steps = test_steps[2:-4]
    # # print(test_steps)
    # print("test_steps.shape:", test_steps.shape)
    # # test_rewards = np.array(test_rewards[:-1]).T
    # print(len(test_rewards))
    # # print([len(xxx) for xxx in test_rewards])
    # test_rewards = np.array(test_rewards)
    # test_rewards = test_rewards[sorted_idx, ...]
    # test_rewards = test_rewards.T
    # test_rewards = test_rewards[:, 2:-4]
    # print("test_rewards.shape:", test_rewards.shape)
    # g = sns.tsplot(time=test_steps/1e4, data=test_rewards, color=colors[iii], condition=labels[iii])
    ##############################################################################################################
    if vis_what == "test":
        # print("new_test_steps:", new_test_steps)
        # new_test_rewards = np.array(new_test_rewards)  # (2, 100)
        # print("\tnew_test_rewards:", new_test_rewards.shape)
        # g = sns.tsplot(time=xnew/1e4, data=new_test_rewards, color=colors[iii], condition=labels[iii])

        final_test_steps = sorted(new_test_steps[0] + new_test_steps[1] + new_test_steps[2])
        finalest_steps = []
        final_test_rewards = []
        for kkk in range(len(final_test_steps)):
            if kkk not in finalest_steps:
                finalest_steps.append(kkk)
                print(kkk, len(final_test_steps))
                one_step = final_test_steps[kkk]
                for one_seed in range(3):
                    if one_step in new_test_steps[one_seed]:
                        # print(new_test_rewards[one_seed][new_test_steps[0].index(one_step)])
                        final_test_rewards.append(
                            new_test_rewards[one_seed][new_test_steps[one_seed].index(one_step)]
                        )
                        break
        finalest_steps = np.array(finalest_steps)
        final_test_rewards = np.array(final_test_rewards).T
        # print("finalest_steps:", finalest_steps.shape)
        # print("final_test_rewards:", final_test_rewards.shape)
        g = sns.tsplot(time=finalest_steps/1e4,
                       data=final_test_rewards,
                       color=colors[iii], condition=labels[iii])

        # fff_test_rewards = []
        # for each_testing_time in range(50):
        #     test_ynew10 = np.interp(xnew, finalest_steps, final_test_rewards[each_testing_time])
        #     if smooth_alg == "tensorb":
        #         test_ynew10 = _tensorboard_smoothing(test_ynew10, weight, remain_last)
        #     if smooth_alg == "nearest":
        #         test_ynew10 = _nearest_smoothing(test_ynew10, weight, remain_last)  #
        #     fff_test_rewards.append(test_ynew10)
        #
        # fff_test_rewards = np.array(fff_test_rewards)
        # print("fff_test_rewards:", fff_test_rewards.shape)
        # g = sns.tsplot(# time=finalest_steps/1e4,
        #                time=xnew/1e4,
        #                data=fff_test_rewards,
        #                color=colors[iii], condition=labels[iii])

    ##############################################################################################################

    # print(test_steps.shape)
    # print(test_rewards.shape)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ TEST ↑

    # g.legend_.remove()




ax.set_ylabel("Episode return")
ax.set_xlabel("Number of frames ($ \\times 10^4$)")
# ax.set_title(f"{domain}+{weather}")
# ax.set_ylim([min_reward_ylim, max_reward_ylim])
ax.set_xlim([min_step_xlim, max_step_lim])
ax.set_xticks(np.linspace(min_step_xlim, max_step_lim, step_span, endpoint=True))

# ax[0].set_yticks(fontproperties='Times New Roman', size=14)
# ax[0].set_xticks(fontproperties='Times New Roman', size=14)

# ax[0].legend(fontsize=10, loc="upper left")
#
#
# ax[1].legend(fontsize=10, loc="upper left")
# ax[1].set_ylabel("Episode return", fontsize=14)
# ax[1].set_xlabel("Number of frames ($ \\times 10^4$)", fontsize=12)
# ax[1].set_title(f"{domain}+{weather}: test")
# ax[1].set_xlim([min_step_xlim, max_step_lim])
# ax[1].set_xticks(np.linspace(min_step_xlim, max_step_lim, step_span, endpoint=True))
# ax[1].set_ylim([min_reward_ylim, max_reward_ylim])

plt.savefig(f"./reward/{domain}-{weather}.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
