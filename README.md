# Luxai-s2-Baseline
Welcome to the Lux AI Challenge Season 2! This repository serves as a baseline for the Lux AI Challenge Season 2, designed to provide participants with a strong starting point for the competition. Our goal is to provide you with a clear, understandable, and modifiable codebase so you can quickly start developing your own AI strategies.

This baseline includes an implementation of the PPO reinforcement learning algorithm, which you can use to train your own agent from scratch. The codebase is designed to be easy to modify, allowing you to experiment with different strategies, reward functions, and other parameters.

In addition to the main training script, we also provide additional tools and resources, including scripts for evaluating your AI strategy, as well as useful debugging and visualization tools. We hope these tools and resources will help you develop and improve your AI strategy more effectively.

More information about the Lux AI Challenge can be found on the competition page: https://www.kaggle.com/competitions/lux-ai-season-2

We look forward to seeing how your AI strategy performs in the competition!

# Getting started
To begin, create a conda environment and activate it using the following commands:
```
conda env create -f environment.yml
conda activate luxai_s2
```
Once the environment is set up, you can start training your agent using the provided training script:
```
python train.py
```
This script will train an agent using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The agent will continuously learn strategies based on the training data.

You can monitor the training process and view various metrics using TensorBoard:
```
tensorboard --logdir runs
```
If you wish to use the behavior cloning imitation learning method, which allows agents to learn strategies from pre-recorded games, you can use the following command to train from a JSON file corresponding to game replays:
```
python train_bc.py
```
Once your agent is trained, you can have it compete in a match and generate a replay of the match using the following command:
```
luxai-s2 path/to/your/main.py path/to/enemy/main.py -v 2 -o replay.html
```
The trained model will be saved in the 'runs' folder. Please ensure to modify the path in the main script to correctly point to your saved model before running the game.

# Train stronger agents
1.**Modify reinforcement learning algorithm.** 

The baseline algorithm *train.py* uses the ppo algorithm in the cleanrl library (https://github.com/vwxyzjn/cleanrl). You can use other reinforcement learning algorithms to try to train stronger agents.

2.**Modify the given way of reward.** 

You can help agents better learn policies by modifying the reward acquisition method or parameters. For example, you can give higher reward to resource collection so that agents can learn to collect more resources. You can modify the default parameters of the reward function in the *impl_config.py* file.

3.**Modify the features of observation.** 

In *parsers*, you can customize and modify the feature and the generation method of reward. There may be some redundant features in the baseline, or some features are not considered. You can add and delete features according to your ideas.

4.**Modify the backbone.** 

The network's backbone adopts the resnet structure, which can be found at *policy/net.py*. You can use a more complex or simpler network structure to modify the backbone.

# Directory Structure
## Description

- `environment.yml`: Specifies the dependencies and packages required to run the code.

- `impl_config.py`: Configuration settings for the policy implementations.

## Files

- `main.py`: Main script for evaluating the baseline.

- `player.py`: Player class implementation.

- `replay.py`: Replay class implementation.

- `train.py`: Training script.

- `train_bc.py`: Training script for behavioral cloning.

- `utils.py`: Utility functions used throughout the code.

## Directories

- `kaggle_replays`: Contains JSON replay files from Kaggle competitions.

- `luxs`: Contains code related to the Lux environment and game mechanics.
  - `cargo.py`: Code for handling cargo units.
  - `config.py`: Configuration settings for the Lux environment.
  - `factory.py`: Factory-related code for unit production.
  - `forward_sim.py`: Code for forward simulation of game states.
  - `kit.py`: Code for handling the game kit.
  - `load_from_replay.py`: Code for loading data from replay files.
  - `team.py`: Code for managing teams in the game.
  - `unit.py`: Code for managing units in the game.
  - `utils.py`: Utility functions used throughout the code.


- `parsers`: Contains parsers for game features and rewards.
  - `__init__.py`: Initialization file for the parsers package.
  - `action_parser_full_act.py`: Parser for full action space.
  - `dense2_reward_parser.py`: Parser for dense reward (version 2).
  - `dense_reward_parser.py`: Parser for dense reward (version 1).
  - `feature_parser.py`: Parser for game features.
  - `sparse_reward_parser.py`: Parser for sparse reward.


- `policy`: Contains the main policy implementation.
  - `__init__.py`: Initialization file for the policy package.
  - `actor_head.py`: Actor head implementation.
  - `algorithm`: Contains the implementation of the algorithm used in the policy.
    - `torch_lux_multi_task_ppo_algorithm_impl.py`: Implementation of the Torch-based multi-task PPO algorithm.
  - `beta_binomial.py`: Implementation of the beta binomial distribution.
  - `impl`: Contains different policy implementations.
    - `multi_task_softmax_policy_impl.py`: Implementation of the multi-task softmax policy.
    - `no_sampling_policy_impl.py`: Implementation of the no-sampling policy.
  - `net.py`: Neural network implementation for the policy.

- `main.py`: Main script for running the baseline.

- `impl_config.py`: Configuration settings for the policy implementations.
