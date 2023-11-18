
# DMR

This repository contains the official code from ''DMR: Decomposed Multi-Modality Representations for Frames and Events Fusion in Visual Reinforcement Learning''


## Abstract
We explore visual reinforcement learning (RL) using two complementary visual modalities: 
frame-based RGB camera and event-based Dynamic Vision Sensor (DVS). 
Existing multi-modality visual RL methods often encounter challenges 
in effectively extracting task-relevant information from multiple modalities 
while suppressing the increased noise, only using indirect reward signals 
instead of pixel-level supervision. To tackle this, we propose a Decomposed 
Multi-Modality Representation (**_DMR_**) framework for visual RL. It explicitly 
decomposes the inputs into three distinct components: combined task-relevant 
features (**_co-features_**), **_RGB-specific noise_**, and **_DVS-specific noise_**. The co-features 
represent the full information from both modalities that is relevant to the RL task; 
the two noise components, each constrained by a data reconstruction loss to avoid information leak, 
are contrasted with the co-features to maximize their difference. Extensive experiments demonstrate that, 
by explicitly separating the different types of information, our approach achieves substantially improved 
policy performance compared to state-of-the-art approaches.

**The overview of DMR learning framework**:
<div align=center>
    <img src="https://github.com/kyoran/DMR/blob/main/vendors/framework.png" 
        alt="framework" width="95%"/>
    <br>
    <b>Fig 1. Illustration of the proposed framework</b>
</div>


## Several typical visual examples

<div align=center>
    <img src="https://github.com/kyoran/DMR/blob/main/vendors/motivation.png" 
        alt="framework" width="45%"/>
    <br>
    <b>Fig 2. Frames and events based RL.</b>
</div>

hello world

<div style="display: flex; align-items: center;">>
    <img src="https://github.com/kyoran/DMR/blob/main/vendors/motivation.png" 
        alt="framework" style="width: 45%; height: auto; margin-right: 20px;"
    />
    <p>Fig 2. Frames and events based RL 123 123 123 12 321 3213 12 312 321 312 321 321 321 213 213.</p>
</div>

## Repository requirements
- create python environment using conda:
```shell
conda create -n carla-py37 python=3.7 -y
conda activate carla-py37
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U gym==0.17.3 cloudpickle==1.5.0 numba==0.51.2 wincertstore==0.2 tornado==4.5.3 msgpack-python==0.5.6 msgpack-rpc-python==0.4.1 stable-baselines3==0.8.0 opencv-python==4.7.0.72 imageio[ffmpeg]==2.28.0 dotmap==1.3.30 termcolor==2.3.0 matplotlib==3.5.3 seaborn-image==0.4.4 scipy==1.7.3 info-nce-pytorch==0.1.4 spikingjelly cupy-cuda117 scikit-image tensorboard kornia timm einops -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- download Carla-0.9.13 from https://github.com/carla-simulator/carla/releases

- unzip Carla and install pythonAPI via:
```shell
cd carla_root_directory/PythonAPI/carla/dist
pip install carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl
```


## DMR training & evaluation
- running CARLA by using:
```shell
DISPLAY= ./CarlaUE4.sh -opengl -RenderOffScreen -carla-rpc-port=12121  # headless mode
```

- running DMR by using:
```shell
bash auto_run_batch_modal.sh
```

- key parameters:
  - --selected_scenario: 