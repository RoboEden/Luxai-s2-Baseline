# Luxai-s2-Baseline
Welcome to the Lux AI Challenge Season 2! This repository is the a baseline of Lux AI Challenge Season 2. The competition page is https://www.kaggle.com/competitions/lux-ai-season-2

# Getting started
To create a conda environment and use it, run
```
conda env create -f environment.yml
conda activate luxai_s2
```
Once installed successfully, you can train your agent by running
```
python train.py
```
After running, the agent will continuously learn strategies from the trajectory through the **PPO** reinforcement learning algorithm. You can use tensorboard to view the changes in various indicators during the training process
```
tensorboard --logdir runs
```
If you want to use the **behavior cloning** imitation learning method to let agents learn strategies from videos, you can run the following code to learn from the json file corresponding to videos
```
python train_bc.py
```
You can use the following command to have the trained agent engage in a match and generate a video of the match:
```
luxai-s2 path/to/your/main.py path/to/enemy/main.py -v 2 -o replay.html
```
The model of the train will be saved in the runs folder. You need to modify the path in the main to the correct path before reading the model.

# Train stronger agents
1.**Modify reinforcement learning algorithm.** 
The baseline algorithm *train.py* uses the ppo algorithm in the cleanrl library (https://github.com/vwxyzjn/cleanrl). You can use other reinforcement learning algorithms to try to train stronger agents.

2.**Modify the given way of reward.** 
You can help agents better learn policies by modifying the reward acquisition method or parameters. For example, you can give higher reward to resource collection so that agents can learn to collect more resources. You can modify the default parameters of the reward function in the *impl_config.py* file.

3.**Modify the features of observation** 
In *parsers*, you can customize and modify the feature and the generation method of reward. There may be some redundant features in the baseline, or some features are not considered. You can add and delete features according to your ideas.

4.**Modify the backbone** 
The network's backbone adopts the resnet structure, which can be found at *policy/net.py*. You can use a more complex or simpler network structure to modify the backbone.
