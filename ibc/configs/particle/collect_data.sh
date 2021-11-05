#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=200 \
 --policy=particle_green_then_blue \
 --task=PARTICLE \
 --dataset_path=ibc/data/particle/2d_oracle_particle.tfrecord \
 --replicas=10  \
 --use_image_obs=False
