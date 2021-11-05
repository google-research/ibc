#!/bin/bash

# Use name of d4rl env as first arg

CMD='python3 ibc/ibc/train_eval.py '
GIN='ibc/ibc/configs/d4rl/mlp_mdn.gin'
DATA="train_eval.dataset_path='ibc/data/d4rl_trajectories/$1/*.tfrecord'"

$CMD -- \
  --alsologtostderr \
  --gin_file=$GIN \
  --task=$1 \
  --tag=mdn \
  --add_time=True \
  --gin_bindings=$DATA
  # not currently calling --video because rendering is broken in the docker?
