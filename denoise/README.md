# Events Temporal Up-sampling

The number of valid events directly affects the performance of event-based tasks, such as reconstruction, detection, and recognition. However, when in low-brightness or slow-moving scenes, events are often sparse and accompanied by noise, which poses challenges for event-based tasks. To solve these challenges, we propose an event temporal up-sampling algorithm to generate more effective and reliable events. Experimental results show that up-sampling events can provide more effective information and improve the performance of downstream tasks, such as improving the quality of reconstructed images and increasing the accuracy of object detection.

![motivation](https://github.com/XIJIE-XIANG/Event-Temporal-Up-sampling/blob/main/data/motivation.png)

For more details, please read our paper "[Temporal Up-Sampling for Asynchronous Events](https://ieeexplore.ieee.org/abstract/document/9858934/)".

## Introduction
Generate up-sampling events on the correct motion trajectory, which includes estimating the motion trajectory of the events by contrast maximization algorithm and up-sampling the events by the temporal point processes (Hawkes Process for main events, Self-correcting Process for noise).


## Code
[main.py](https://github.com/XIJIE-XIANG/Event-Temporal-Up-sampling/blob/main/main.py): up-sampling events

[Contrast_Maximization.py](https://github.com/XIJIE-XIANG/Event-Temporal-Up-sampling/blob/main/Contrast_Maximization.py): estimate event motion trajectory

[Temporal_Point_Processes.py](https://github.com/XIJIE-XIANG/Event-Temporal-Up-sampling/blob/main/Temporal_Point_Processes.py): up-sampling events by Hawkes Process and Self-correcting Process

[event_process.py](https://github.com/XIJIE-XIANG/Event-Temporal-Up-sampling/blob/main/event_process.py): including warp events, save up-sampling events, show result, etc.


## Usage
Change event_path in [main.py](https://github.com/XIJIE-XIANG/Event-Temporal-Up-sampling/blob/main/main.py) to your own path.


## Dependencies
python=3.8


## Publication
If you use this code in an academic context, please cite the following publication:

X. Xiang, L. Zhu, J. Li, Y. Tian and T. Huang, "Temporal Up-Sampling for Asynchronous Events," 2022 IEEE International Conference on Multimedia and Expo (ICME), 2022, pp. 01-06.


>@INPROCEEDINGS{Xiang22ICME,  
>  author={Xiang, Xijie and Zhu, Lin and Li, Jianing and Tian, Yonghong and Huang, Tiejun},  
>  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},   
>  title={Temporal Up-Sampling for Asynchronous Events},   
>  year={2022},  
>  pages={01-06}}




