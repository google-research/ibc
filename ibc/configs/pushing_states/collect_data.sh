#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=200 \
 --policy=oracle_push \
 --task=PUSH \
 --dataset_path=ibc/data/block_push_states_location/oracle_push.tfrecord \
 --replicas=10  \
 --use_image_obs=False
