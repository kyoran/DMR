
# DMR

This repository contains the official code from ''DMR: Decomposed Multi-Modality Representations for Frames and Events Fusion in Visual Reinforcement Learning''

- [2022/11/18]: DMR is currently under review for CVPR 2024.


## Abstract
We explore visual reinforcement learning (RL) using two complementary visual modalities: 
**_frame-based RGB camera_** and **_event-based Dynamic Vision Sensor (DVS)_**. 
Existing multi-modality visual RL methods often encounter challenges 
in effectively extracting task-relevant information from multiple modalities 
while suppressing the increased noise, only using indirect reward signals 
instead of pixel-level supervision. To tackle this, we propose a Decomposed 
Multi-Modality Representation (**_DMR_**) framework for visual RL. It explicitly 
decomposes the inputs into three distinct components: combined task-relevant 
features (**_co-features_**), **_RGB-specific noise_**, and **_DVS-specific noise_**. The co-features 
represent the full information from both modalities that is relevant to the RL task; 
the two noise components, each constrained by a data reconstruction loss to avoid information leak, 
are contrasted with the co-features to maximize their difference.

**The overview of DMR learning framework**:
<div align=center>
    <img src="https://github.com/kyoran/DMR/blob/main/vendors/framework.png" 
        alt="framework" width="95%"/>
</div>


## Several typical visual examples

- Illustration of our motivation
<div align="center">
<table>
<tr>
    <td align="center" width="55%"><img src="https://github.com/kyoran/DMR/blob/main/vendors/motivation.png" 
        alt="motivation"/>
    </td>
    <td align="left" width="45%">
        (i) In the first row, insufficient ambient light causes RGB underexposure, leading to the overlooking of the front pedestrian and resulting in a forward policy aligned with the lane direction that could cause collisions.
        <br/>   <br/>     
        (ii) In the second row, the lack of texture in DVS causes the person and the background to blend, leading to a left-turn policy to avoid the highlighted area on the right.
        <br/>   <br/> 
        (iii) In contrast, our method (third row) can fully take advantage of RGB and DVS to extract task-relevant information and eliminate task-irrelevant and noisy information through joint TD and DMR learning, thereby obtaining an optimal evasion policy. 
    </td>
</tr>
</table>
</div>

- Illustration of decomposition capability of DMR
<div align="center">
<table>
<tr>
    <td align="center" width="20%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/case/rgb.png"/>
    </td>
    <td align="center" width="20%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/case/dvs.png"/>
    </td>
    <td align="left" width="60%" rowspan="3">
        (i) First row depicts the original observations and corresponding CAMs of DMR. 
            In the extremely low-light condition, DVS can capture the front pedestrian 
            while RGB camera suffers from exposure failure.
        <br/><br/><br/>
        (ii) It can be seen from second row that RGB noise highlights the high beam region on the road, 
            while DVS noise is activated across a broader region, with the highest activation on the building.
        <br/><br/><br/>
        (iii) The co-features in the third row attentively grasp the pedestrian and the right roadside 
              simultaneously, which are crucial for driving decision-making.
    </td>
</tr>
<tr>
    <td align="center" width="20%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/case/rgb-noise.png"/>
    </td>
    <td align="center" width="20%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/case/dvs-noise.png"/>
    </td>
</tr>
<tr>
    <td align="center" width="20%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/case/co-rgb.png"/>
    </td>
    <td align="center" width="20%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/case/co-dvs.png"/>
    </td>
</tr>
</table>
<br>

</div>


- A long sequence demonstration
<div align="center">
<table>
<tr>
    <td align="center">Time</td>
    <td align="center">RGB Frame</td>
    <td align="center">DVS Events</td>
    <td align="center">RGB Noise</td>
    <td align="center">DVS Noise</td>
    <td align="center">Co-features<br>on RGB</td>
    <td align="center">Co-features<br>on DVS</td>
</tr>
<tr>
    <td align="center" width="10%">Time #1</td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/1-rgb.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/1-dvs.jpg"/>
    </td>
    <td align="center" width="15%">
          <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/1-rgb-noise.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/1-dvs-noise.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/1-rgb-co.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/1-dvs-co.jpg"/>
    </td>
</tr>
<tr>
    <td align="center" width="10%">Time #2</td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/2-rgb.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/2-dvs.jpg"/>
    </td>
    <td align="center" width="15%">
          <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/2-rgb-noise.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/2-dvs-noise.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/2-rgb-co.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/2-dvs-co.jpg"/>
    </td>
</tr>
<tr>
    <td align="center" width="10%">Time #3</td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/3-rgb.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/3-dvs.jpg"/>
    </td>
    <td align="center" width="15%">
          <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/3-rgb-noise.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/3-dvs-noise.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/3-rgb-co.jpg"/>
    </td>
    <td align="center" width="15%">
        <img src="https://github.com/kyoran/DMR/blob/main/vendors/sequence/3-dvs-co.jpg"/>
    </td>
</tr>
</table>
<br>
</div>

The table above illustrates a vehicle with high beam headlights approaching from 
a distance to near in the opposite lane at three different time instances, Time #1, #2, and #3. 
It is clear that the **_RGB noise_** emphasizes the vehicle's high beam headlights and the buildings on the right, 
whereas the **_DVS noise_** focuses on the dense event region on the right. 
Both types of noise contain a substantial amount of task-irrelevant information, covering unnecessary broad areas. 
In contrast, the **_co-features_** generates a more focused area that is relevant for RL by excluding irrelevant regions.
These areas precisely cover the vehicle on the opposite lane and the right roadside,
which are crucial cues for driving policies.

The variations in Class Activation Mapping (CAM) closely mirror the alterations in the real scene throughout the entire process.
When the vehicle approaches, the RGB noise broadens due to illumination changes, and the co-features focus more on the vehicle. 
In co-features, there is also a gradual increase in emphasis on the left roadside, and the CAM uniformly cover the right roadside.


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

- choices of some key parameters in `train_testm.py`:
  - selected_scenario: 'jaywalk', 'highbeam'
  - selected_weather: 'midnight', 'hard_rain'
  - perception_type: 
    - single-modality perception: 'RGB-Frame', 'DVS-Frame', 'Depth-Frame', 'DVS-Voxel-Grid', 'LiDAR-BEV', 
    - multi-modality perception: 'RGB-Frame+DVS-Frame', 'RGB-Frame+DVS-Voxel-Grid', 'RGB-Frame+Depth-Frame', 'RGB-Frame+LiDAR-BEV'
  - encoder_type:
    - single-modality encoder: 'pixelCarla098'
    - multi-modality encoder: 'DMR_CNN', 'DMR_SNN', 'pixelCrossFusion', 'pixelEFNet', 'pixelFPNNet', 'pixelRENet', ...