#!/bin/bash
# Launches training or oracle eval.
# See ibc/ibc/README.md for usage examples.

source gbash.sh || exit 1

# Choose config.
DEFINE_array task --type=string --delim="," "PUSH" "Choose tasks."
DEFINE_array gin_bindings --type=string --delim="," "" "Additional gin bindings."
DEFINE_bool rm_logdir false "Whether to rm previous logdir."
DEFINE_bool use_debug_data true "Whether to use debug data when training locally, else use --dataset_glob."
DEFINE_string mode "" "can be: {'train', 'eval'}."

# Training arguments.
DEFINE_string train_gin_file "mlp_ebm.gin" "Choose a gin config file from configs/."
DEFINE_string train_dataset_glob "/tmp/xarm_data-0000*.tfrecord" "Data to use for training."
DEFINE_string train_tag "name_this_experiment" "Tag to use for experiments."

# Eval arguments.
DEFINE_int eval_num_episodes 200 "Number of evaluation episodes (per replica)."
DEFINE_string eval_policy "" "Which policy to use for eval (e.g. oracle_push)."
DEFINE_string eval_dataset_path "" "Where to store the rollout data."
DEFINE_int eval_replicas 1 "How many replicas to use when doing eval."
DEFINE_bool eval_use_image_obs false "If True, store image observations in output dataset."

gbash::init_google "$@"

set -uexo pipefail

ROOT=ibc
readonly TRAIN_GIN_FILE=$ROOT/ibc/configs/${FLAGS_train_gin_file}

task_flags=()
for item in "${FLAGS_task[@]}"; do
  task_flags+=("--task=$item")
done

gin_bindings=()
for item in "${FLAGS_gin_bindings[@]}"; do
  gin_bindings+=("--gin_bindings=\"$item\"")
done
echo "${gin_bindings[@]}"

RUN_CMD () {
  echo "$1"
  eval "$1" || { echo "ERROR: '$1' failed!" ; exit 1 ; }
}

readonly script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
readonly outer_dir="${script_dir}/../.."
readonly current_python_path="${PYTHONPATH:-}"
readonly updated_python_path="${current_python_path}:${outer_dir}"
train_eval_exec="PYTHONPATH=${updated_python_path} python3 ${script_dir}/train_eval.py "
policy_eval_exec="PYTHONPATH=${updated_python_path} python3 ${script_dir}/../data/policy_eval.py "


cd "${outer_dir}"

if [[ "$FLAGS_mode" == "train" ]]; then

  if (( FLAGS_rm_logdir )); then
    echo "Deleting previous local training dir."
    rm -rf /tmp/ebm
  fi
  # Which local data to train on.
  if (( FLAGS_use_debug_data )); then
    LOCAL_DATA=$ROOT/test_data/${FLAGS_task}/*.tfrecord
    echo "Using debug data at $LOCAL_DATA."
  else
    LOCAL_DATA="${FLAGS_train_dataset_glob}"
    echo "Using your data at $LOCAL_DATA."
  fi

  RUN_CMD "${train_eval_exec} \
  --gin_file=\"${TRAIN_GIN_FILE}\" \
  ${task_flags[@]} \
  --skip_eval=False \
  --tag=${FLAGS_train_tag} \
  --add_time=True \
  --gin_bindings=\"train_eval.root_dir='/tmp/ebm'\" \
  --gin_bindings=\"train_eval.batch_size=4\" \
  --gin_bindings=\"train_eval.eval_interval=4\" \
  --gin_bindings=\"train_eval.eval_episodes=2\" \
  --gin_bindings=\"ImplicitBehavioralCloningAgent.num_counter_examples=8\" \
  --gin_bindings=\"train_eval.dataset_path='${LOCAL_DATA}'\" \
  ${gin_bindings[@]} \
  --alsologtostderr"

elif [[ "$FLAGS_mode" == "eval" ]]; then

  RUN_CMD "${policy_eval_exec} \
  ${task_flags[@]} \
  --num_episodes=${FLAGS_eval_num_episodes} \
  --policy=${FLAGS_eval_policy} \
  --replicas=${FLAGS_eval_replicas} \
  --use_image_obs=${FLAGS_eval_use_image_obs} \
  --dataset_path=${FLAGS_eval_dataset_path} \
  ${gin_bindings[@]} \
  --alsologtostderr"

else

  echo "See ibc/ibc/README.md for usage examples."

fi
