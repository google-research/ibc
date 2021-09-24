# Implicit Behavioral Cloning

This codebase contains the official implementation of the Implicit Behavioral Cloning (IBC) algorithm from the paper:

***Florence et al., [Implicit Behavioral Cloning (arxiv link)](https://arxiv.org/abs/2109.00137), Conference on Robotic Learning (CoRL) 2021.***

<img src="docs/energy_pop_teaser.png"/>

## Abstract

We find that across a wide range of robot policy learning scenarios, treating supervised policy learning with an implicit model generally performs better, on average, than commonly used explicit models. We present extensive experiments on this finding, and we provide both intuitive insight and theoretical arguments distinguishing the properties of implicit models compared to their explicit counterparts, particularly with respect to approximating complex, potentially discontinuous and multi-valued (set-valued) functions. On robotic policy learning tasks we show that implicit behavioral cloning policies with energy-based models (EBM) often outperform common explicit (Mean Square Error, or Mixture Density) behavioral cloning policies, including on tasks with high-dimensional action spaces and visual image inputs. We find these policies provide competitive results or outperform state-of-the-art offline reinforcement learning methods on the challenging human-expert tasks from the D4RL benchmark suite, despite using no reward information. In the real world, robots with implicit policies can learn complex and remarkably subtle behaviors on contact-rich tasks from human demonstrations, including tasks with high combinatorial complexity and tasks requiring 1mm precision.

## Prerequisites

The code for this project uses python 3.7+ and the following pip packages:

```
python -m pip install --upgrade pip
pip install \
  absl-py==0.12.0 \
  gin-config==0.4.0 \
  matplotlib==3.4.3 \
  mediapy==1.0.3 \
  pybullet==3.1.6 \
  tensorflow==2.6.0 \
  tensorflow-probability==0.13.0 \
  tf-agents-nightly==0.10.0.dev20210914 \
  tqdm==4.62.2
```

For mujoco / D4RL support, you also need to install:

```
pip install \
  mujoco_py==2.0.2.5 \
  git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

Notd that the above will require that you have a local mujoco installed (see
[here](https://github.com/openai/mujoco-py) for installation details).

It's also recommended to run all unit tests before launching training:

```
cd <path_to>/ibc
./run_tests.sh
```

## Dataset Generation and Training

See [ibc/README.md](ibc/README.md) for details.

## Citation

If you found our paper/code useful in your research, please consider citing:

```
@article{
  author = {Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, Jonathan Tompson},
  title = {Implicit Behavioral Cloning},
  journal = {Conference on Robotic Learning (CoRL)},
  month = {November},
  year = {2021},
}
```
