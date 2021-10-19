#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/pushing_pixels/pixel_mdn_best.gin \
  --task=PUSH \
  --tag=pixel_mdn_best \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/block_push_visual_location/oracle_*.tfrecord'" \
  --video
