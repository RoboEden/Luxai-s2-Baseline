import json
import numpy as np
import torch
import gym
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
    """
    计算给定player_id的平均回报。
    
    参数:
        info_list: 包含sub_rewards信息的列表
        player_id: 需要计算平均回报的玩家ID
    
    返回:
        player_id的平均回报
    """
    total_sum = 0
    num_elements = len(info_list)
    
    # 遍历info列表中的每个元素
    for element in info_list:
        # 提取sub_rewards并求和
        total_sum += sum(element['agents'][player_id]['sub_rewards'].values())
    
    # 计算均值
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
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super(LuxRecordEpisodeStatistics, self).step(
            action
        )
        self.episode_returns += sum(rewards)
        self.episode_lengths += 1
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )

def make_env(seed):
    def thunk():
        # env = gym.make(env_id)
        env = LuxEnv()
        env = LuxRecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk