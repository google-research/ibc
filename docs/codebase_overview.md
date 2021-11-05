## Codebase Overview

The highest level structure contains:


- `ibc/`
    - `data/` <-- tools to generate datasets, and feed data for training
    - `environments/` <-- a collection of environments
    - `networks/` <-- TensorFlow models for state inputs and/or vision inputs
    - ...

The above directories are algorithm-agnostic, and the implementation of specific algorithms
are mostly in:

- `ibc/ibc/`
    - `agents/` <-- holds the majority of the BC algorithm details, including:
        - `ibc_agent.py` <-- class for IBC training
        - `ibc_policy.py` <-- class for IBC inference
        - `mcmc.py` <-- implements optimizers used for IBC training/inference
        - similar files for MSE and MDN policies.
    - `losses/` <-- loss functions
        - `ebm_loss.py` <-- several different EBM-style loss functions.
        - `gradient_loss.py` <-- gradient penalty for Langevin
    - `configs/` <-- configurations for different trainings (including hyperparams)
    - ... other various utils for making training and evaluation happen.

A couple more notes for you the reader:

1. The codebase was optimized for large-scale experimentation and trying out many different ideas.  With hindsight it could be much simpler to implement a simplified version of only the core essentials.
2. The codebase heavily uses TF Agents, so we don't have to re-invent various wheels, and it is recommended you take a look at the Guide to get a sense: https://www.tensorflow.org/agents/overview
