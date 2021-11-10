# Implicit Behavioral Cloning

This codebase contains the official implementation of the *Implicit Behavioral Cloning (IBC)* algorithm from our paper:



**Implicit Behavioral Cloning [(website link)](https://implicitbc.github.io/)  [(arXiv link)](https://arxiv.org/abs/2109.00137)** </br>
*Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, Jonathan Tompson* </br>
Conference on Robot Learning (CoRL) 2021

![](./docs/insert.gif)  |  ![](./docs/sort.gif)
:-------------------------:|:-------------------------:|

<img src="docs/energy_pop_teaser.png"/>

## Abstract

We find that across a wide range of robot policy learning scenarios, treating supervised policy learning with an implicit model generally performs better, on average, than commonly used explicit models. We present extensive experiments on this finding, and we provide both intuitive insight and theoretical arguments distinguishing the properties of implicit models compared to their explicit counterparts, particularly with respect to approximating complex, potentially discontinuous and multi-valued (set-valued) functions. On robotic policy learning tasks we show that implicit behavioral cloning policies with energy-based models (EBM) often outperform common explicit (Mean Square Error, or Mixture Density) behavioral cloning policies, including on tasks with high-dimensional action spaces and visual image inputs. We find these policies provide competitive results or outperform state-of-the-art offline reinforcement learning methods on the challenging human-expert tasks from the D4RL benchmark suite, despite using no reward information. In the real world, robots with implicit policies can learn complex and remarkably subtle behaviors on contact-rich tasks from human demonstrations, including tasks with high combinatorial complexity and tasks requiring 1mm precision.

## Prerequisites

The code for this project uses python 3.7+ and the following pip packages:

```bash
python3 -m pip install --upgrade pip
pip install \
  absl-py==0.12.0 \
  gin-config==0.4.0 \
  matplotlib==3.4.3 \
  mediapy==1.0.3 \
  opencv-python==4.5.3.56 \
  pybullet==3.1.6 \
  scipy==1.7.1 \
  tensorflow==2.6.0 \
  keras==2.6.0 \
  tf-agents==0.11.0rc0 \
  tqdm==4.62.2
```

(Optional): For Mujoco support, see [`docs/mujoco_setup.md`](docs/mujoco_setup.md).  Recommended to skip it
unless you specifically want to run the Adroit and Kitchen environments.

## Quickstart: from 0 to a trained IBC policy in 10 minutes.

**Step 1**: Install listed Python packages above in  [Prerequisites](#Prequisites).

**Step 2**: Run unit tests (should take less than a minute), and do this from the directory *just above the top-level `ibc` directory*:

```bash
./ibc/run_tests.sh
```

**Step 3**: Check that Tensorflow has GPU access:

```bash
python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```

If the above prints `False`, see the following requirements, notably CUDA 11.2 and cuDNN 8.1.0: https://www.tensorflow.org/install/gpu#software_requirements.

**Step 4**: Let's do an example Block Pushing task, so first let's **download oracle data** (or see [Tasks](#tasks) for how to generate it):

```bash
cd ibc/data
wget https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip
unzip block_push_states_location.zip && rm block_push_states_location.zip
cd ../..
```

**Step 5**: Set PYTHONPATH to include the directory *just above top-level `ibc`*, so if you've been following the commands above it is:

```bash
export PYTHONPATH=$PYTHONPATH:${PWD}
```

**Step 6**: On that example Block Pushing task, we'll next do a **training + evaluation** with Implicit BC:

```bash
./ibc/ibc/configs/pushing_states/run_mlp_ebm.sh
```

*Some notes*:

- On an example single-GPU machine (GTX 2080 Ti), the above trains at about 18 steps/sec, and should get to high success rates in 5,000 or 10,000 steps (roughly 5-10 minutes of training).
- The `mlp_ebm.gin` is just one config, which is meant to be reasonably fast to train, with only 20 evals at each interval, and is not suitable for all tasks.  See [Tasks](#tasks) for more configs.
- Due to the `--video` flag above, you can watch a video of the learned policy in action at: `/tmp/ibc_logs/mlp_ebm/ibc_dfo/`... navigate to the `videos/ttl=7d` subfolder, and by default there should be one example `.mp4` video saved every time you do an evaluation interval.

**(Optional) Step 7**: For the pybullet-based tasks, we also have real-time interactive visualization set up through a visualization server, so in one terminal:

```bash
cd <path_to>/ibc/..
export PYTHONPATH=$PYTHONPATH:${PWD}
python3 -m pybullet_utils.runServer
```

And in a different terminal run the oracle a few times with the `--shared_memory` flag:

```bash
cd <path_to>/ibc/..
export PYTHONPATH=$PYTHONPATH:${PWD}
python3 ibc/data/policy_eval.py -- \
  --alsologtostderr \
  --shared_memory \
  --num_episodes=3 \
  --policy=oracle_push \
  --task=PUSH
```

**You're done with Quickstart!**  See below for more [Tasks](#tasks), and also see [`docs/codebase_overview.md`](docs/codebase_overview.md) and [`docs/workflow.md`](docs/workflow.md) for additional info.




## Tasks

### Task: Particle

In this task, the goal is for the agent (black dot) to first go to the green dot, then the blue dot.

Example IBC policy  | Example MSE policy
:-------------------------:|:-------------------------:
![](./docs/particle_langevin_10000.gif)  |  ![](./docs/particle_mse_10000.gif) |

#### Get Data

We can either generate data from scratch, for example for 2D (takes 15 seconds):

```bash
./ibc/ibc/configs/particle/collect_data.sh
```

Or just download all the data for all different dimensions: <a name="particle-data"></a>

```bash
cd ibc/data/
wget https://storage.googleapis.com/brain-reach-public/ibc_data/particle.zip
unzip particle.zip && rm particle.zip
cd ../..
```

#### Train and Evaluate

Let's start with some small networks, on just the 2D version since it's easiest to visualize, and compare MSE and IBC.  Here's a small-network (256x2) IBC-with-Langevin config, where `2` is the argument for the environment dimensionality.

<!--  partial verified: 96% success, 10k steps, 50 episodes evaluated, 13.3 steps/sec  -->
```bash
./ibc/ibc/configs/particle/run_mlp_ebm_langevin.sh 2
```

And here's an idenitcally sized network (256x2) but with MSE config:

<!--  partial verified: 5% success, 10k steps, 20 episodes evaluated, 21.7 steps/sec  -->
```bash
./ibc/ibc/configs/particle/run_mlp_mse.sh 2
```

For the above configurations, we suggest comparing the rollout videos, which you can find at `/tmp/ibc_logs/...corresponding_directory../videos/`. At the top of this section is shown a comparison at 10,000 training steps for the two different above configs.


And here are the **best configs** respectfully for **IBC** (with langevin) and **MSE**, in this case run on the 16-dimensional environment: <a name="particle-train"></a>

```
./ibc/ibc/configs/particle/run_mlp_ebm_langevin_best.sh 16
./ibc/ibc/configs/particle/run_mlp_mse_best.sh 16
```

Note: the *`_best`* config is kind of slow for Langevin to train, but even just `./ibc/ibc/configs/particle/run_mlp_ebm_langevin.sh 16` (smaller network) seems to solve the 16-D environment pretty well, and is much faster to train.



### Task: Block Pushing (from state observations)

#### Get Data

We can either generate data from scratch (~2 minutes for 2,000 episodes: 200 each across 10 replicas):

```bash
./ibc/ibc/configs/pushing_states/collect_data.sh
```

Or we can download data from the web:<a name="pushing-states-data"></a>

```bash
cd ibc/data/
wget https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip
unzip 'block_push_states_location.zip' && rm block_push_states_location.zip
cd ../..
```

#### Train and Evaluate

Here's reasonably fast-to-train config for *IBC with DFO*:

<!--  partial verified: 100% in 10k steps, 18 steps/sec -->
```bash
./ibc/ibc/configs/pushing_states/run_mlp_ebm.sh
```

Or here's a config for *IBC with Langevin*:

<!--  partial verified: 95% in 5k steps, 6.5 steps/sec -->
```bash
./ibc/ibc/configs/pushing_states/run_mlp_ebm_langevin.sh
```

Or here's a comparable, reasonably fast-to-train config for *MSE*:

<!--  partial verified: 85% in 10k steps, 18 steps/sec -->
```bash
./ibc/ibc/configs/pushing_states/run_mlp_mse.sh
```

Or to run the **best configs** respectfully **for IBC, MSE, and MDN** (some of these might be slower to train than the above): <a name="pushing-states-train"></a>

<!--  partial verified: 100% at 15k steps, 18 steps/sec -->
<!--  partial verified: 87% at 15k steps, 18 steps/sec -->
<!--  partial verified: 75% at 5k steps, 18 steps/sec -->
```bash
./ibc/ibc/configs/pushing_states/run_mlp_ebm_best.sh
./ibc/ibc/configs/pushing_states/run_mlp_mse_best.sh
./ibc/ibc/configs/pushing_states/run_mlp_mdn_best.sh
```

### Task: Block Pushing (from image observations)

#### Get Data

Download data from the web: <a name="pushing-pixels-data"></a>

```bash
cd ibc/data/
wget https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_visual_location.zip
unzip 'block_push_visual_location.zip' && rm block_push_visual_location.zip
cd ../..
```

#### Train and Evaluate

Here is an *IBC with Langevin* configuration which should actually converge faster than the IBC-with-DFO that we reported in the paper:

<!--  partial verified: 100% at 10k steps, 6.5 steps/sec, at 90x120 w/ 128 batch-->
<!--  partial verified: 100% at 5k steps, 4.1 steps/sec, at 180x240 w/ 128 batch-->
```bash
./ibc/ibc/configs/pushing_pixels/run_pixel_ebm_langevin.sh
```

And here are the **best configs** respectfully for **IBC** (with DFO), **MSE**, and **MDN**: <a name="pushing-pixels-train"></a>

<!-- partial verified: 94% at 10k steps, 8.0 steps/sec, 180x240 w/ 128 batch-->
<!-- partial verified: 68% at 10k steps, 9.0 steps/sec, 180x240 w/ 128 batch -->
<!-- partial verified: 94% at 15k steps, 9.0 steps/sec, 90x120 w/ 128 batch -->
```bash
./ibc/ibc/configs/pushing_pixels/run_pixel_ebm_best.sh
./ibc/ibc/configs/pushing_pixels/run_pixel_mse_best.sh
./ibc/ibc/configs/pushing_pixels/run_pixel_mdn_best.sh
```


### Task: D4RL Adroit and Kitchen

#### Get Data

The D4RL human demonstration training data used for the paper submission can be downloaded using the commands below.  This data has been processed into a `.tfrecord` format from the original D4RL data format: <a name="d4rl-data"></a>

```bash
cd ibc/data && mkdir -p d4rl_trajectories && cd d4rl_trajectories
wget https://storage.googleapis.com/brain-reach-public/ibc_data/door-human-v0.zip \
     https://storage.googleapis.com/brain-reach-public/ibc_data/hammer-human-v0.zip \
     https://storage.googleapis.com/brain-reach-public/ibc_data/kitchen-complete-v0.zip \
     https://storage.googleapis.com/brain-reach-public/ibc_data/kitchen-mixed-v0.zip \
     https://storage.googleapis.com/brain-reach-public/ibc_data/kitchen-partial-v0.zip \
     https://storage.googleapis.com/brain-reach-public/ibc_data/pen-human-v0.zip \
     https://storage.googleapis.com/brain-reach-public/ibc_data/relocate-human-v0.zip
unzip '*.zip' && rm *.zip
cd ../../..
```

### Run Train Eval:


Here are the **best configs** respectfully for **IBC** (with Langevin), and **MSE**: <a name="d4rl-train"></a>
On a 2080 Ti GPU test, this IBC config trains at only 1.7 steps/sec, but it is about 10x faster on TPUv3.


<!--  partial verified: 2704.5 avg return on pen, 10k steps, 100 episodes evaluated, 1.7 steps/sec  -->
<!--  partial verified: 1660.4 avg return on pen, 10k steps, 100 episodes evaluated, 25.5 steps/sec  -->

```bash
./ibc/ibc/configs/d4rl/run_mlp_ebm_langevin_best.sh pen-human-v0
./ibc/ibc/configs/d4rl/run_mlp_mse_best.sh pen-human-v0
```

The above commands will run on the `pen-human-v0` environment, but you can swap this arg for whichever of the provided Adroit/Kitchen environments.

Here also is an MDN config you can try.  The network size is tiny but if you increase it heavily then it seems to get NaNs during training. In general MDNs can be finicky.  A solution should be possible though.

```
./ibc/ibc/configs/d4rl/run_mlp_mdn.sh pen-human-v0
```

## Summary for Reproducing Results

For the tasks that we've been able to open-source, results from the paper should be reproducible by using the linked data and command-line args below.

| Task  | Figure/Table in paper | Data | Train + Eval commands |
| --- | --- | --- | --- |
| Coordinate regression  | Figure 4  | See colab | See colab |
| D4RL Adroit + Kitchen  | Table 2 | [Link](#d4rl-data) | [Link](#d4rl-train) |
| N-D particle  | Figure 6 | [Link](#particle-data) | [Link](#particle-train) |
| Simulated pushing, single target, states  | Table 3 | [Link](#pushing-states-data) | [Link](#pushing-states-train) |
| Simulated pushing, single target, pixels | Table 3 | [Link](#pushing-pixels-data) | [Link](#pushing-pixels-train) |


## Citation

If you found our paper/code useful in your research, please consider citing:

```
@article{florence2021implicit,
    title={Implicit Behavioral Cloning},
    author={Florence, Pete and Lynch, Corey and Zeng, Andy and Ramirez, Oscar and Wahid, Ayzaan and Downs, Laura and Wong, Adrian and Lee, Johnny and Mordatch, Igor and Tompson, Jonathan},
    journal={Conference on Robot Learning (CoRL)},
    month = {November},
    year={2021}
}
```
