#!/bin/bash

set -eu

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=3 \
 --replicas=2  \
 --policy=particle_green_then_blue \
 --task=PARTICLE \
 --use_image_obs=False \
 --dataset_path=/tmp/ibc_tmp/data \
 --output_path=/tmp/ibc_tmp/vid \
 --video
