#!/bin/bash

## Use "N" of the N-d environment as the arg

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/particle/mlp_ebm_langevin.gin \
  --task=PARTICLE \
  --tag=langevin \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/particle/$1d_oracle_particle*.tfrecord'" \
  --gin_bindings="ParticleEnv.n_dim=$1" \
  --video
