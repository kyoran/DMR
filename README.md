
# DM3DP: A Decomposed Multi-Modal Markov Decision Process for Visuomotor Reinforcement Learning

## 1st. Install environment from scratch

### install cuda and cudnn (optional)
- install cuda 11.7.0 from: https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local
```shell
# download cuda 11.7.0
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
# export environmental variables of cuda 11.7:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
export PATH=$PATH:/usr/local/cuda-11.7/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```

- download cudnn 8.5.0 from: https://developer.nvidia.com/rdp/cudnn-archive
```shell
# copy cudnn file to the filesystem and chmod
tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -p cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```


### create python environment using conda

- install anaconda3
```shell

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Linux-x86_64.sh (install anaconda3 by following the instructions)
```
- install necessary packages
```shell
conda create -n carla-py37 python=3.7
conda activate carla-py37
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python==4.7.0.72 gym==0.19.0 imageio==2.28.0 dotmap==1.3.30 termcolor==2.3.0 \
 matplotlib==3.5.3 scipy==1.7.3 info-nce-pytorch==0.1.4 tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- testing environment 
```python
# testing environment 
import torch
from torch.backends import cudnn
a = torch.tensor(1.).cuda()
print(cudnn.is_available())
print(cudnn.is_acceptable(a.cuda()))
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```


### download CARLA-0.9.13
```shell
cd ~/carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz

tar -zxvf CARLA_0.9.13.tar.gz
tar -zxvf AdditionalMaps_0.9.13.tar.gz

cd PythonAPI/carla/dist
pip install carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl
```

- running CARLA by using:
```shell
DISPLAY= ./CarlaUE4.sh -opengl -RenderOffScreen -carla-rpc-port=12121
```

*Note*: if your operation system do not support opengl, try using:
```shell
sudo apt-get update && sudo apt-get install -y --fix-missing xserver-xorg mesa-utils libvulkan1 libomp5 # install vulkan first
DISPLAY= ./CarlaUE4.sh -vulkan -nosound -RenderOffscreen -carla-rpc-port=12121
```

## 2nd. running baseline methods or ours

```shell
chmod +x auto_run_batch_modal.sh
./auto_run_batch_modal.sh
```
