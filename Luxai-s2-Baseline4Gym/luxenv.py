from collections.abc import Callable, Iterable, Mapping
from typing import Any
import random
import numpy as np
from impl_config import EnvParam
import gymnasium as gym
from gymnasium import spaces
import tree
from luxai_s2.env import EnvConfig, LuxAI_S2
from parsers import ActionParser,FeatureParser,DenseRewardParser,Dense2RewardParser,SparseRewardParser
from luxs.kit import obs_to_game_state
from replay import random_init
from player import Player

from torch import Tensor
import torch

from gymnasium.vector.utils import concatenate, create_empty_array
from copy import deepcopy

def torch2np(x):
    if isinstance(x, torch.Tensor):
        return x[0].detach().cpu().numpy()
    else:
        return x


def get_action_space(rule_based_early_step, map_size):
    action_space = spaces.Dict(
        {
            0: get_single_action_space(rule_based_early_step, map_size), 
            1: get_single_action_space(rule_based_early_step, map_size)
        }
    )
    return action_space


def get_single_action_space(rule_based_early_step, map_size):
    type_space = np.full((map_size, map_size), 7)
    direction_space = np.full((map_size, map_size), 5)
    resource_space = np.full((map_size, map_size), 5)
    amount_space = np.full((map_size, map_size), 10)
    repeat_space = np.full((map_size, map_size), 2)
    n_space = np.full((map_size, map_size), 20)
    unit_space = np.stack([type_space, direction_space, resource_space, amount_space, repeat_space, n_space])

    action_space = {
        "factory_act": spaces.MultiDiscrete(np.full((map_size, map_size), 4), dtype=np.float64), 
        "unit_act": spaces.MultiDiscrete(unit_space, dtype=np.float64), 
    }
    if not rule_based_early_step:
        action_space.update(
            {
                "bid" : spaces.Discrete(11), 
                "factory_spawn": spaces.Dict(
                    {
                        "location": spaces.Discrete(map_size*map_size), 
                        "water": spaces.Box(low=0, high=1, shape=()), 
                        "metal": spaces.Box(low=0, high=1, shape=())
                    }
                )
            }
        )

    action_space = spaces.Dict(action_space)
    return action_space

def get_observation_space(map_size):
    obs_space = spaces.Dict(
        {
            'player_0': get_single_observation_space(map_size), 
            'player_1': get_single_observation_space(map_size)
        }
    )
    return obs_space

def get_single_observation_space(map_size):
    global_feature_names = ['env_step',                 # 1000+10
                            'cycle',                    # 20
                            'hour',                     # 50
                            'daytime_or_night',         # 2
                            'num_factory_own',          # 5
                            'num_factory_enm',          # 5
                            'total_lichen_own',         # 48 * 48 * 100
                            'total_lichen_enm',         # 48 * 48 * 100
                            'factory_total_power_own',  # 9999
                            'factory_total_ice_own',    # 9999
                            'factory_total_water_own',  # 9999
                            'factory_total_ore_own',    # 9999
                            'factory_total_metal_own',  # 9999
                            'num_light_own',            # 9999
                            'num_heavy_own',            # 9999
                            'robot_total_power_own',    # 9999
                            'robot_total_ice_own',      # 9999
                            'robot_total_water_own',    # 9999
                            'robot_total_ore_own',      # 9999
                            'robot_total_metal_own',    # 9999
                            'factory_total_power_enm',  # 9999
                            'factory_total_ice_enm',    # 9999
                            'factory_total_water_enm',  # 9999
                            'factory_total_ore_enm',    # 9999
                            'factory_total_metal_enm',  # 9999
                            'num_light_enm',            # 9999
                            'num_heavy_enm',            # 9999
                            'robot_total_power_enm',    # 9999
                            'robot_total_ice_enm',      # 9999
                            'robot_total_water_enm',    # 9999
                            'robot_total_ore_enm',      # 9999
                            'robot_total_metal_enm']    # 9999
    global_feature_space = [
        1000, 
        20, 
        50, 
        2, 
        5, 
        5, 
        230400,
        230400, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999
    ]
    global_feature_space = spaces.MultiDiscrete(np.array(global_feature_space), dtype=np.float64)

    map_featrue_names = {
        'ice': 9999, 
        'ore': 9999, 
        'rubble': 100, 
        'lichen': 100, 
        'lichen_strains': 11, # -1, 0, 1
        'lichen_strains_own': 2, 
        'lichen_strains_enm': 2, 
        'valid_region_indicator': 2, 
        'factory_id': 10, 
        'factory_power': 9999, 
        'factory_ice': 9999, 
        'factory_water': 9999, 
        'factory_ore': 9999, 
        'factory_metal': 9999, 
        'factory_own': 2, 
        'factory_enm': 2, 
        'factory_can_build_light': 2, 
        'factory_can_build_heavy': 2, 
        'factory_can_grow_lichen': 2, 
        'factory_water_cost': 9999, 
        'unit_id': 9999, 
        'unit_power': 9999, 
        'unit_ice': 9999, 
        'unit_water': 9999, 
        'unit_ore': 9999, 
        'unit_metal': 9999, 
        'unit_own': 2, 
        'unit_enm': 2, 
        'unit_light': 2, 
        'unit_heavy': 2
    }
    map_featrue_space = np.tile(np.array(list(map_featrue_names.values())).reshape(30, 1, 1), (1, map_size, map_size))
    map_featrue_space = spaces.MultiDiscrete(map_featrue_space, dtype=np.float64)

    action_feature_space = spaces.Dict(
        {
            'unit_indicator': spaces.MultiDiscrete(np.full((map_size, map_size), 2), dtype=np.float64),
            'type': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 7), dtype=np.float64),
            'direction': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 5), dtype=np.float64),
            'resource': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 5), dtype=np.float64),
            'amount': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 10), dtype=np.float64),
            'repeat': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 2), dtype=np.float64),
            'n': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 9999), dtype=np.float64)
        }
    )
    obs_space = spaces.Dict(
        {
            'global_feature': global_feature_space, 
            'map_feature': map_featrue_space, 
            'action_feature': action_feature_space
        }
    )
    return obs_space

def crop_map(env, pos=(0, 0), size=24):
    (x, y) = pos
    H, W = size, size

    def _crop(arr):
        return arr[x:x + H, y:y + W]

    env.state.env_cfg.map_size = size

    board = env.state.board
    board.width = W
    board.height = H
    board.env_cfg.map_size = H
    board.lichen = _crop(board.lichen)
    board.lichen_strains = _crop(board.lichen_strains)
    board.factory_occupancy_map = _crop(board.factory_occupancy_map)
    board.valid_spawns_mask = _crop(board.valid_spawns_mask)

    board.valid_spawns_mask[[0, -1], :] = False
    board.valid_spawns_mask[:, [0, -1]] = False

    board.map.ice = _crop(board.map.ice)
    board.map.ore = _crop(board.map.ore)
    board.map.rubble = _crop(board.map.rubble)
    board.map.height = H
    board.map.width = W

class LuxEnv(gym.Env):
    
    def __init__(self, kaggle_replays=None, **kwargs):
        super().__init__(**kwargs)

        self.proxy = LuxAI_S2(
            collect_stats=True,
            verbose=False,
            MAX_FACTORIES=EnvParam.MAX_FACTORIES,
        )
        self.env_cfg = self.proxy.state.env_cfg
        self.agents = {}
        self.players = []
        if EnvParam.parser == 'sparse':
            self.reward_parser = SparseRewardParser()
        elif EnvParam.parser == 'dense':
            self.reward_parser = DenseRewardParser()
        elif EnvParam.parser == 'dense2':
            self.reward_parser = Dense2RewardParser()
        else:
            raise NotImplementedError
        self.feature_parser = FeatureParser()
        self.action_parser = ActionParser()
        self.game_state = [{}, {}]
        self.kaggle_replays = kaggle_replays

        self.rule_based_early_step = EnvParam.rule_based_early_step
        
        self.action_space = get_action_space(self.rule_based_early_step, self.env_cfg.map_size)
        self.observation_space = get_observation_space(self.env_cfg.map_size)
        self.single_vas_space = self.get_va_space(self.env_cfg.map_size)
    def reset_proxy(self, seed=None, options=None):
        assert not ((EnvParam.init_from_replay_ratio != 0.) and (EnvParam.map_size != EnvConfig.map_size))
        seed = seed if seed is not None else np.random.SeedSequence().generate_state(1)
        if self.kaggle_replays and EnvParam.init_from_replay_ratio != 0:
            if np.random.rand() < EnvParam.init_from_replay_ratio:
                self.proxy.load_from_replay = True
                random_init(self.proxy, self.kaggle_replays)
            else:
                self.proxy.load_from_replay = False
                self.proxy.reset(seed=seed, options=options)
        elif EnvParam.map_size != EnvConfig.map_size:
            self.load_from_replay = False
            while True:
                self.proxy.env_cfg.map_size = EnvConfig.map_size
                obs = self.proxy.reset(seed=seed, options=options)
                if EnvParam.map_size < self.proxy.env_cfg.map_size:
                    pos = np.random.randint(0, self.proxy.env_cfg.map_size - EnvParam.map_size, size=2)
                    crop_map(self.proxy, pos, size=EnvParam.map_size)
                if self.proxy.state.board.ice.sum() >= self.proxy.state.board.factories_per_team * 2:
                    break
        else:
            self.proxy.load_from_replay = False
            self.proxy.reset(seed=seed, options=options)
        obs = self.proxy.state.get_obs()
        obs = {agent: obs for agent in self.proxy.agents}
        self.real_obs = obs
        return obs

    def seed(self, seed):
        self.proxy.seed(seed)

    def reset(self, seed=None, options=None):
        obs = self.reset_proxy(seed=seed, options=options)
        self.agents = {player: Player(player, self.env_cfg) for player in self.proxy.agents}
        self.players = list(self.agents.keys())
        if self.rule_based_early_step:
            while self.proxy.state.real_env_steps < 0:
                actions = {}
                for player in self.proxy.agents:
                    o = obs[player]
                    a = self.agents[player].early_setup(self.proxy.env_steps, o)
                    actions[player] = a
                obs, rewards, terminations, truncations, infos = self.proxy.step(actions)
        else:
            dones = {"player_0": False, "player_1": False}
        for player_id, player in enumerate(self.proxy.agents):
            o = obs[player]
            self.game_state[player_id] = obs_to_game_state(self.proxy.env_steps, self.env_cfg, o)
        obs_list, global_info = self.feature_parser.parse(obs, env_cfg=self.env_cfg)
        self.reward_parser.reset(self.game_state, global_info, self.proxy.state.stats)

        return obs_list, global_info

    def step(self, actions):
        actions, action_stats = self.action_parser.parse(self.game_state, actions)
        obs, rewards, terminations, truncations, infos = self.proxy.step(actions)  # interact with env
        dones = {key: terminations[key] or truncations[key] for key in terminations.keys()}
        self.real_obs = obs
        for player_id, player in enumerate(self.proxy.agents):
            o = obs[player]
            self.game_state[player_id] = obs_to_game_state(self.proxy.env_steps, self.env_cfg, o)
        obs_list, global_info = self.feature_parser.parse(obs, env_cfg=self.env_cfg)
        reward, sub_rewards = self.reward_parser.parse(
            dones,
            self.game_state,
            self.proxy.state.stats,
            global_info,
        )  # reward parser
        env_stats_logs = self.feature_parser.log_env_stats(self.proxy.state.stats)
        # done = dones["player_0"] or dones["player_1"]
        info = {"agents": [], "episodes": []}
        for team in [0, 1]:
            agent_info = {
                "stats": global_info[f"player_{team}"],
                "env_stats": env_stats_logs,
                "sub_rewards": sub_rewards[team],
                "step": self.proxy.state.real_env_steps,
                "action_stats": action_stats[team]
            }
            info["agents"].append(agent_info)
        return obs_list, reward, terminations, truncations, info
    
    def eval(self, own_policy, enemy_policy):
        np2torch = lambda x, dtype: torch.tensor(x).type(dtype).cuda()
        own_id = random.randint(0, 1)
        enemy_id = 1 - own_id

        obs_list = self.reset()
        done = False
        episode_length = 0
        return_own = 0
        return_enemy = 0
        while not done:
            actions = {}
            for id, policy in zip([own_id, enemy_id], [own_policy, enemy_policy]):
                valid_action = self.get_valid_actions(id)
                _, _, raw_action, _ = policy(
                    np2torch([obs_list['player_'+str(id)]['global_feature']], torch.float32),
                    np2torch([obs_list['player_'+str(id)]['map_feature']], torch.float32), 
                    tree.map_structure(lambda x: np2torch([x], torch.int16), obs_list['player_'+str(id)]['action_feature']),
                    tree.map_structure(lambda x: np2torch([x], torch.bool), valid_action)
                )
                actions[id] = raw_action
            actions = tree.map_structure(lambda x: torch2np(x), actions)                
            obs_list, reward, done, info = self.step(actions)
            return_own += reward[own_id]
            return_enemy += reward[enemy_id]
            episode_length += 1
        return episode_length, return_own, return_enemy

    def get_valid_actions(self, player_id):
        return self.action_parser.get_valid_actions(self.game_state[player_id], player_id)
    
    def get_va_space(self, map_size):
        space = {
            'factory_act': spaces.MultiBinary([4, map_size, map_size]), 
            'move': spaces.MultiBinary([5, 2, map_size, map_size]), 
            'transfer': spaces.MultiBinary([5, 5, 2, map_size, map_size]), 
            'pickup': spaces.MultiBinary([5, 2, map_size, map_size]), 
            'dig': spaces.MultiBinary([2, map_size, map_size]), 
            'self_destruct': spaces.MultiBinary([2, map_size, map_size]), 
            'recharge': spaces.MultiBinary([2, map_size, map_size]), 
            'do_nothing': spaces.MultiBinary([map_size, map_size])  
        }
        if not self.rule_based_early_step:
            space.update({'bid': spaces.MultiBinary(11), 'factory_spawn': spaces.MultiBinary([map_size, map_size])})
        space = spaces.Dict(
            space
        )
        
        return space
    
    def concatenate_obs(self, observations_list):
        bs = len(observations_list)
        observation_shape_each_player = get_single_observation_space()
        output_obs = create_empty_array(
            observation_shape_each_player, n=bs, fn=np.zeros
        )
        observations = concatenate(
                observations_list, output_obs, observation_shape_each_player
            )
        return deepcopy(observations)
    
    def concatenate_action(self, action_list):
        bs = len(action_list)
        action_shape_each_player = get_single_action_space(self.rule_based_early_step, self.env_cfg.map_size)
        output_action = create_empty_array(
            action_shape_each_player, n=bs, fn=np.zeros
        )
        actions = concatenate(
                action_list, output_action, action_shape_each_player
            )
        return deepcopy(actions)

    def concatenate_va(self, valid_action_list):
        bs = len(valid_action_list)
        output_action = create_empty_array(
            self.single_vas_space, n=bs, fn=np.zeros
        )
        actions = concatenate(
                valid_action_list, output_action, self.single_vas_space
            )
        return deepcopy(actions)

def lux_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    # try:
    while True:
        command, data = pipe.recv()
        if command == "reset":
            observation = env.reset()
            pipe.send((observation, True))
        elif command == "step":
            observation, reward, termination, truncation, info = env.step(data)
            done = {key: termination[key] or truncation[key] for key in termination.keys()}
            done = done["player_0"] or done["player_1"]
            if done:
                observation, _ = env.reset()
            pipe.send(((observation, reward, termination, truncation, info), True))
        elif command == "seed":
            env.seed(data)
            pipe.send((None, True))
        elif command == "get_valid_actions":
            valid_actions = env.get_valid_actions(data)
            pipe.send((valid_actions))
        elif command == "close":
            pipe.send((None, True))
            break
        elif command == "_check_observation_space":
            pipe.send((data == env.observation_space, True))
        elif command == "_check_spaces":
            pipe.send(
                (
                    (data[0] == env.observation_space, data[1] == env.action_space),
                    True,
                )
            )
        elif command == "eval":
            episode_length, r_own, r_enemy = env.eval(data[0], data[1])
            pipe.send(episode_length, r_own, r_enemy)
        else:
            raise RuntimeError(
                f"Received unknown command `{command}`. Must "
                "be one of {`reset`, `step`, `seed`, `close`, "
                "`_check_observation_space`, `get_valid_actions`, "
                "`_check_spaces`, "
                "`eval`}."
            )
    # except (KeyboardInterrupt, Exception):
    #     error_queue.put((index,) + sys.exc_info()[:2])
    #     pipe.send((None, False))
    # finally:
    env.close()

class LuxSyncVectorEnv(gym.vector.AsyncVectorEnv):
    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True, shared_memory=False, worker=lux_worker):
        super().__init__(env_fns=env_fns, observation_space=observation_space, action_space=action_space, copy=copy, shared_memory=shared_memory, worker=worker)
        dummy_env = env_fns[0]()
        self.single_vas_space = dummy_env.get_va_space(dummy_env.env_cfg.map_size)
        self.dummy_env_cfg = deepcopy(dummy_env.env_cfg)
        self.vas = create_empty_array(
            self.single_vas_space, n=self.num_envs, fn=np.zeros
        )
        dummy_env.close()
        del dummy_env

        self.rule_based_early_step = EnvParam.rule_based_early_step
    
    def get_valid_actions(self, player_id):
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(("get_valid_actions", player_id))
        valid_actions = [pipe.recv() for pipe in self.parent_pipes]
        # valid_actions = [env.get_valid_actions(player_id) for env in self.envs]
        self.vas = concatenate(
            self.single_vas_space, valid_actions, self.vas
        )
        return deepcopy(self.vas)
    
    def split(self, action):
        actions = [{} for _ in range(self.num_envs)]
        for key, value in action.items():
            if isinstance(value, Tensor):
                value = value.cpu().detach().numpy()
            if isinstance(value, dict):
                sub_action = self.split(value)
                value = sub_action
            for _value, _action in zip(value, actions):
                _action.update({key: _value})
        return actions
    
    def concatenate_obs(self, observations_list):
        bs = len(observations_list)
        observation_shape_each_player = get_single_observation_space(self.dummy_env_cfg.map_size)
        output_obs = create_empty_array(
            observation_shape_each_player, n=bs, fn=np.zeros
        )
        observations = concatenate(
                observation_shape_each_player, observations_list, output_obs
            )
        return deepcopy(observations)

    def concatenate_action(self, action_list):
        bs = len(action_list)
        action_shape_each_player = get_single_action_space(self.rule_based_early_step, self.dummy_env_cfg.map_size)
        output_action = create_empty_array(
            action_shape_each_player, n=bs, fn=np.zeros
        )
        actions = concatenate(
                action_shape_each_player, action_list, output_action
            )
        return deepcopy(actions)

    def concatenate_va(self, valid_action_list):
        bs = len(valid_action_list)
        output_action = create_empty_array(
            self.single_vas_space, n=bs, fn=np.zeros
        )
        actions = concatenate(
                self.single_vas_space, valid_action_list, output_action
            )
        return deepcopy(actions)
    
    def eval(self, eval_policy, enemy_policy=None):
        self._assert_is_running()
        if enemy_policy is None:
            enemy_policy = eval_policy
        for pipe in self.parent_pipes:
            pipe.send(("eval", (eval_policy, enemy_policy)))
        results = [pipe.recv() for pipe in self.parent_pipes]
        results = self._process_eval_resluts(results)

        return deepcopy(results)
    
    def _process_eval_resluts(self, results):
        if self.num_envs==1:
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
