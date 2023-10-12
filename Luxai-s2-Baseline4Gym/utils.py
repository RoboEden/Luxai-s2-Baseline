import json
import numpy as np
import torch
# import gym
import gymnasium as gym
from luxenv import LuxEnv
import time
from collections import deque

def save_args(args, file_path):
    args_dict = vars(args)
    with open(file_path, 'w') as file:
        json.dump(args_dict, file, indent=4)

def save_model(net, file_path):
    torch.save(net.state_dict(), file_path)

def load_model(net, file_path):
    net.load_state_dict(torch.load(file_path))
    return net

def eval_model(net):
    env = LuxEnv()
    eval_results = env.eval(net, net)
    env.close()
    return eval_results

def _process_eval_resluts(results):
    if len(results)==1:
        results = {
            "avg_episode_length": results[0][0], 
            "avg_return_own": results[0][1], 
            "avg_return_enemy": results[0][2]
        }
    else:
        results = np.array(results)
        episode_length, return_own, return_enemy = results[:, 0], results[:, 1], results[:, 2]
        results = {
            "avg_episode_length": np.mean(episode_length), 
            "std_episode_length": np.std(episode_length), 
            "avg_return_own": np.mean(return_own), 
            "std_return_own": np.std(return_own), 
            "avg_return_enemy": np.mean(return_enemy), 
            "std_return_enemy": np.std(return_enemy)
        }
    return results

def cal_mean_return(info_list, player_id):
    total_sum = 0
    num_elements = len(info_list)
    for element in info_list:
        total_sum += sum(element[player_id]['sub_rewards'].values())
    
    return total_sum / num_elements if num_elements > 0 else 0

class LuxRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(LuxRecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super(LuxRecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(2, dtype=np.float32)
        self.episode_lengths = np.zeros(2, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminations, truncations, infos = super(LuxRecordEpisodeStatistics, self).step(
            action
        )
        dones = list({key:terminations[key] or truncations[key] for key in terminations.keys()}.values())
        self.episode_returns += rewards
        self.episode_lengths += 1
        # if not self.is_vector_env:
        #     infos = [infos]
        #     dones = [dones]
        for i in range(len(dones)):
            if dones[i]:
                infos['agents'][i] = infos['agents'][i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos["episodes"].append(episode_info)
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            # dones if self.is_vector_env else dones[0],
            terminations,
            truncations,
            # infos if self.is_vector_env else infos[0],
            infos
        )

def make_env(seed,replay_dir):
    def thunk():
        env = LuxEnv(replay_dir)
        env = LuxRecordEpisodeStatistics(env)
        env.seed(seed)
        return env

    return thunk