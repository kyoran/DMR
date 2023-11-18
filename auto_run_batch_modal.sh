#!/bin/bash

SUIT=carla

DOMAIN=highbeam
#DOMAIN=jaywalk
#WEATHER=midnight
WEATHER=hard_rain

AGENT=deepmdp
#AGENT=sac
#AGENT=mlr

#PERCEPTION=RGB-Frame
#PERCEPTION=DVS-Frame
#PERCEPTION=DVS-Stream
#PERCEPTION=DVS-Voxel-Grid
#PERCEPTION=RGB-Frame+DVS-Frame
#PERCEPTION=RGB-Frame+DVS-Voxel-Grid
PERCEPTION=RGB-Frame+LiDAR-BEV

#ENCODER=pixelCarla098
#ENCODER=eVAE

#ENCODER=pixelInputFusion
#ENCODER=pixelCrossFusion
#ENCODER=pixelCatSep
#ENCODER=pixelCat
#ENCODER=pixelEFNet
#ENCODER=pixelRENet
#ENCODER=pixelFPNNet
#ENCODER=pixelConNewV4
#ENCODER=pixelConNewV4_Repel
ENCODER=DMR_CNN
#ENCODER=DMR_SNN


DECODER=identity

#RPC_PORT=12121
#RPC_PORT=12232
RPC_PORT=12343
#RPC_PORT=12454

#TM_PORT=19121
#TM_PORT=19232
TM_PORT=19343
#TM_PORT=19454

CUDA_DEVICE=2

#SEED=111
SEED=222
#SEED=333
#SEED=444
#SEED=999


UNIQUE_ID=${SUIT}+${DOMAIN}+${WEATHER}+${AGENT}+${PERCEPTION}+${ENCODER}+${SEED}
LOGFILE=./logs/${UNIQUE_ID}.log
WORKDIR=~/logs/${UNIQUE_ID}


echo ${UNIQUE_ID}
mkdir -p ${WORKDIR}


CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -u train_testm.py \
	--work_dir ${WORKDIR} \
	--suit carla \
	--domain_name ${DOMAIN} \
	--selected_weather ${WEATHER} \
	--agent ${AGENT} \
	--perception_type ${PERCEPTION} \
	--encoder_type ${ENCODER} \
	--decoder_type ${DECODER} \
	--action_model_update_freq 1 \
	--transition_reward_model_update_freq 1 \
	--carla_rpc_port ${RPC_PORT} \
	--carla_tm_port ${TM_PORT} \
  --carla_timeout 30 \
  --frame_skip 1 \
  --init_steps 1000 \
  --num_train_steps 110000 \
  --max_episode_steps 500 \
  --rl_image_size 128 \
  --num_cameras 1 \
  --action_type continuous \
  --alpha_lr 1e-4 \
  --actor_lr 1e-4 \
  --critic_lr 1e-4 \
  --encoder_lr 1e-4 \
  --decoder_lr 1e-4 \
  --replay_buffer_capacity 10000 \
  --batch_size 32 \
  --EVAL_FREQ_EPISODE 20 \
  --LOG_FREQ 10000 \
  --num_eval_episodes 20 \
  --save_tb \
  --save_model \
  --save_video \
  --do_metrics \
  --seed ${SEED}>${LOGFILE} 2>&1 &


