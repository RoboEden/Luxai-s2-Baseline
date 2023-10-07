# Luxai-s2-Baseline
Welcome to the **Lux AI Challenge Season 2**! This repository serves as a baseline for the Lux AI Challenge Season 2, designed to provide participants with a strong starting point for the competition. Our goal is to provide you with a clear, understandable, and modifiable codebase so you can quickly start developing your own AI strategies.

This baseline includes an implementation of the **PPO** reinforcement learning algorithm, which you can use to train your own agent from scratch. The codebase is designed to be easy to modify, allowing you to experiment with different strategies, reward functions, and other parameters.

In addition to the main training script, we also provide additional tools and resources, including scripts for evaluating your AI strategy, as well as useful debugging and visualization tools. We hope these tools and resources will help you develop and improve your AI strategy more effectively.

More information about the Lux AI Challenge can be found on the competition page: https://www.kaggle.com/competitions/lux-ai-season-2

We look forward to seeing how your AI strategy performs in the competition!

# Getting started with RL
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

Once your agent is trained, you can have it compete in a match and generate a replay of the match using the following command:
```
luxai-s2 path/to/your/main.py path/to/enemy/main.py -v 2 -o replay.html
```
The trained model will be saved in the 'runs' folder. Please ensure to modify the path in the main script to correctly point to your saved model before running the game.

# Learn from replays
If you're interested in employing the behavior cloning method, a type of imitation learning that enables agents to learn strategies from previously played games, you can follow these steps to train from JSON files that correspond to game replays.

To start, visit https://www.kaggle.com/datasets/kaggle/meta-kaggle and download the *Episodes.csv* and *EpisodeAgents.csv* files. Once downloaded, place them in your workspace directory.

To download game replays, you can execute the following command:
```
python download.py
```
This script allows you to modify arguments to download the top-ranking replays as well as customized replays based on your needs. After successfully downloading the JSON file to be learned from, simulate the learning process by running the following command:
```
python train_bc.py
```
This command initiates the behavior cloning training process, enabling you to start learning from your downloaded game replays.

# Train stronger agents
1.**Modify reinforcement learning algorithm.** 

The current baseline algorithm `train.py` employs the ppo algorithm from the cleanrl library (https://github.com/vwxyzjn/cleanrl). However, there is room for improvement by experimenting with other state-of-the-art reinforcement learning algorithms. Consider trying different algorithms to train stronger and more efficient agents.

2.**Refine the reward acquisition method.** 

Enhancing the way agents acquire rewards can significantly impact policy learning. By adjusting the reward mechanism or its parameters, agents can be guided more effectively towards desired behaviors. For instance, assigning higher rewards for resource collection could encourage agents to prioritize this behavior. Tweak the default reward function parameters located in `impl_config.py` to customize the reward system.

3.**Tailor observation features to the task.** 

In `parsers` allows for customization and modification of the observation features and reward generation methods. The current baseline may include redundant or overlooked features. Take the opportunity to add or remove features according to your domain knowledge and insights. This can lead to more informative observations and improved agent performance.

4.**Experiment with network architecture.** 

The current network backbone follows the resnet structure, defined in `policy/net.py`. However, it's worth exploring the impact of different network architectures on agent learning. Consider experimenting with more complex or simpler network structures to find the optimal balance between model capacity and computational efficiency.

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

# Training Curves

If you use the default parameters, the changes in average survival steps and average return are as follows:
![eval_avg_episode_length (1)](https://github.com/RoboEden/Luxai-s2-Baseline/assets/72459814/008536ac-dcca-4869-be81-4f71e58f9c71)
![eval_avg_return_own (1)](https://github.com/RoboEden/Luxai-s2-Baseline/assets/72459814/18c37172-e005-4647-b7e9-caa9a233c625)


While training, the return will steadily increase. However, it may take roughly 2 days to train from scratch to achieve 1000 survival steps, so please maintain your patience.
