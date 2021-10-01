All scripts are expected to be run from the ibc directory. e.g.:

```
cd <path_to>/ibc

# Then run commands below.
```

## Running IBC on block push

### Generate a dataset:

```
./ibc/main.sh \
  --mode=eval \
  --task=PUSH \
  --eval_dataset_path=/tmp/blocks/dataset/oracle_push.tfrecord \
  --eval_num_episodes=200 \
  --eval_policy=oracle_push \
  --eval_replicas=10  \
  --eval_use_image_obs=false
```

### Run Train Eval:

EBM:

```
./ibc/main.sh \
  --mode=train \
  --task=PUSH \
  --gin_bindings="train_eval.root_dir='/tmp/ebm'" \
  --train_dataset_glob="/tmp/blocks/dataset/oracle_push*.tfrecord" \
  --train_gin_file=mlp_ebm_langevin.gin \
  --train_tag=name_this_experiment
```

MDN:

```
./ibc/main.sh \
  --mode=train \
  --task=PUSH \
  --gin_bindings="train_eval.root_dir='/tmp/ebm'" \
  --train_dataset_glob="/tmp/blocks/dataset/oracle_push*.tfrecord" \
  --train_gin_file=mlp_mdn.gin \
  --train_tag=name_this_experiment
```

## Running on particle

### Generate a dataset

To generate lots of data on default 2D, run the following

```
./ibc/main.sh \
  --mode=eval \
  --eval_num_episodes=200 \
  --eval_policy=particle_green_then_blue \
  --task=PARTICLE \
  --eval_dataset_path=/tmp/particle/dataset/oracle_particle.tfrecord \
  --eval_replicas=10  \
  --eval_use_image_obs=False
```


### Run Train Eval:

EBM:

```
./ibc/main.sh \
  --mode=train \
  --task=PARTICLE \
  --gin_bindings="train_eval.root_dir='/tmp/ebm'" \
  --train_dataset_glob="/tmp/particle/dataset/oracle_particle*.tfrecord" \
  --train_gin_file=mlp_ebm.gin \
  --train_tag=name_this_experiment
```

MSE:

```
./ibc/main.sh \
  --mode=train \
  --task=PARTICLE \
  --gin_bindings="train_eval.root_dir='/tmp/ebm'" \
  --train_dataset_glob="/tmp/particle/dataset/oracle_particle*.tfrecord" \
  --train_gin_file=mlp_mse.gin \
  --train_tag=name_this_experiment
```

## Running on D4RL.


### Run Train Eval:

EBM:

```
# pen-human-v0
./ibc/main.sh \
  --mode=train \
  --task=pen-human-v0 \
  --gin_bindings="train_eval.root_dir='/tmp/ebm'" \
  --train_dataset_glob="$(pwd)/data/d4rl_trajectories/pen-human-v0/*.tfrecord" \
  --train_gin_file=mlp_ebm.gin \
  --train_tag=name_this_experiment
```
