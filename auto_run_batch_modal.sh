#!/bin/bash

SUIT=carla

DOMAIN=highbeam
WEATHER=midnight

AGENT=deepmdp

PERCEPTION=RGB-Frame
#PERCEPTION=DVS-Frame
#PERCEPTION=DVS-Stream
#PERCEPTION=RGB-Frame+DVS-Frame
#PERCEPTION=RGB-Frame+DVS-Voxel-Grid

ENCODER=pixelCarla098
#ENCODER=eVAE

#ENCODER=pixelInputFusion
#ENCODER=pixelCatSep
#ENCODER=pixelCat
#ENCODER=pixelConNewV2

DECODER=identity

RPC_PORT=12121
TM_PORT=19121

CUDA_DEVICE=0

SEED=111


UNIQUE_ID=${SUIT}+${DOMAIN}+${WEATHER}+${AGENT}+${PERCEPTION}+${ENCODER}+${SEED}
LOGFILE=./logs/${UNIQUE_ID}.log
WORKDIR=/dev/logs/${UNIQUE_ID}


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
  --alpha_lr 1e-4 \
  --actor_lr 1e-4 \
  --critic_lr 1e-4 \
  --encoder_lr 1e-4 \
  --decoder_lr 1e-4 \
  --replay_buffer_capacity 10000 \
  --batch_size 32 \
  --EVAL_FREQ_EPISODE 20 \
  --LOG_FREQ 10000 \
  --num_eval_episodes 50 \
  --save_tb \
  --do_carla_metrics \
  --seed ${SEED}>${LOGFILE} 2>&1 &


