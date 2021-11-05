# Mujoco setup


For (optional) Mujoco / D4RL support, you will also need additional pre-reqs. You'll need some non-Python pre-reqs:

1. Ensure a local Mujoco installed (see
[here](https://github.com/openai/mujoco-py#install-mujoco) for installation details), you'll need `~/.mujoco/mujoco200` and `~/.mujoco/mjkey.txt`
2. `sudo apt-get install libosmesa6-dev`
3. Install patchelf, for example with [these few commands.](https://github.com/openai/mujoco-py/issues/147#issuecomment-361417560)

And then Python packages:

```bash
pip install mujoco_py==2.0.2.5
```

```
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
# edit the setup.py file as here: https://github.com/rail-berkeley/d4rl/pull/126
pip install -e .
```
Note in our case we needed some symlinks as follows to make various packages happy: `cd ~/.mujoco && sudo ln -s mujoco200_linux mujoco200)`.
