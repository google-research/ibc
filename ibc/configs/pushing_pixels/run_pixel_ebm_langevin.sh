#!/bin/bash

CMD='python3 ibc/ibc/train_eval.py '
GIN='ibc/ibc/configs/pushing_pixels/pixel_ebm_langevin.gin'
DATA="train_eval.dataset_path='ibc/data/block_push_visual_location/oracle_*.tfrecord'"

$CMD -- \
  --alsologtostderr \
  --gin_file=$GIN \
  --task=PUSH \
  --tag=pixel_ibc_langevin \
  --add_time=True \
  --gin_bindings=$DATA \
  --video
