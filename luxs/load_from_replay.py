import json
from pprint import pprint
from copy import deepcopy
import sys
sys.path.append('..')
from parsers import ActionParser,FeatureParser,DenseRewardParser,Dense2RewardParser,SparseRewardParser
from luxai_s2.env import EnvConfig, LuxAI_S2
from luxenv import LuxEnv
from luxs.kit import obs_to_game_state, GameState, Board, json_obs_to_game_state
import gzip
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from luxs.cargo import UnitCargo
from luxs.config import EnvConfig
from luxs.team import Team, FactionTypes
from luxs.unit import Unit
from luxs.factory import Factory

def get_obs_action_from_json(json_dict: dict, max_length=20):
    json_obs_0 = list(map(lambda i: json.loads(i[0]['observation']['obs']), json_dict['steps']))

    json_action_0 = list(map(lambda i: i[0]['action'], json_dict['steps']))
    json_action_1 = list(map(lambda i: i[1]['action'], json_dict['steps']))

    return json_obs_0[:max_length], json_action_0[:max_length], json_action_1[:max_length]


def str_to_pos(key):
    number = key.split(',')
    return int(number[0]), int(number[1])


def change(c: dict, last_obs):
    obs = deepcopy(last_obs)
    for key, value in c.items():
        x, y = str_to_pos(key)
        obs[x][y] = value
    return obs

def check(obs):
    for i, o in enumerate(obs):
        if isinstance(o['player_0']['board']['rubble'], dict):
            if o['player_0']['board']['rubble'] == {}:
                o['player_0']['board']['rubble'] = deepcopy(obs[i-1]['player_0']['board']['rubble'])
            else:
                o['player_0']['board']['rubble'] = change(o['player_0']['board']['rubble'], 
                                                          obs[i-1]['player_0']['board']['rubble'])
        if isinstance(o['player_1']['board']['rubble'], dict):
            o['player_1']['board']['rubble'] = deepcopy(o['player_0']['board']['rubble'])

        if 'ore' not in o['player_0']['board'].keys():
            o['player_0']['board']['ore'] = deepcopy(obs[i-1]['player_0']['board']['ore'])
        if 'ore' not in o['player_1']['board'].keys():
            o['player_1']['board']['ore'] = deepcopy(o['player_0']['board']['ore'])

        if 'ice' not in o['player_0']['board'].keys():
            o['player_0']['board']['ice'] = deepcopy(obs[i-1]['player_0']['board']['ice'])
        if 'ice' not in o['player_1']['board'].keys():
            o['player_1']['board']['ice'] = deepcopy(o['player_0']['board']['ice'])

        if isinstance(o['player_0']['board']['lichen'], dict):
            if o['player_0']['board']['lichen'] == {}:
                o['player_0']['board']['lichen'] = deepcopy(obs[i-1]['player_0']['board']['lichen'])
            else:
                o['player_0']['board']['lichen'] = change(o['player_0']['board']['lichen'], 
                                                          obs[i-1]['player_0']['board']['lichen'])
        if isinstance(o['player_1']['board']['lichen'], dict):
            o['player_1']['board']['lichen'] = deepcopy(o['player_0']['board']['lichen'])
        
        if isinstance(o['player_0']['board']['lichen_strains'], dict):
            if o['player_0']['board']['lichen_strains'] == {}:
                o['player_0']['board']['lichen_strains'] = deepcopy(obs[i-1]['player_0']['board']['lichen_strains'])
            else:
                o['player_0']['board']['lichen_strains'] = change(o['player_0']['board']['lichen_strains'], 
                                                          obs[i-1]['player_0']['board']['lichen_strains'])
        if isinstance(o['player_1']['board']['lichen_strains'], dict):
            o['player_1']['board']['lichen_strains'] = deepcopy(o['player_0']['board']['lichen_strains'])
    return obs

def replay_to_state_action(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    print(f'load json from:{path}')
    proxy = LuxAI_S2(
        collect_stats=True,
        verbose=False,
        MAX_FACTORIES=data['configuration']['env_cfg']['MAX_FACTORIES'],
    )
    feature_parser = FeatureParser()
    action_parser = ActionParser()
    max_length = data['configuration']['episodeSteps']
    json_obs, json_action_0, json_action_1 = get_obs_action_from_json(data, max_length=max_length)
    first_step = 0 - json_obs[0]['real_env_steps'] + 1
    
    actions = dict(player_0=list(),player_1=list())
    obs = list(map(lambda i: {agent: deepcopy(json_obs[i]) for agent in ['player_0', 'player_1']}, range(len(json_obs))))
    obs = check(obs)
    obs_list = list(map(lambda i: feature_parser.json_parser(obs[i], 
                                                             env_cfg=proxy.state.env_cfg),
                         range(first_step, len(json_obs))))
    obs_ = dict(player_0=list(map(lambda i: obs_list[i]['player_0'], range(len(json_obs) - first_step))),
                player_1=list(map(lambda i: obs_list[i]['player_1'], range(len(json_obs) - first_step))))
    actions = list(map(lambda i: action_parser.inv_parse(obs[i], 
                                                         [json_action_0[i], 
                                                          json_action_1[i]], 
                                                          proxy.state.env_cfg), 
                       range(first_step, len(json_obs))))
    actions = dict(player_0=list(map(lambda i: actions[i]['player_0'], range(len(json_obs) - first_step))), 
                   player_1=list(map(lambda i: actions[i]['player_1'], range(len(json_obs) - first_step))))
    valid_action_0 = list(map(lambda i: action_parser.get_valid_actions_from_json(proxy.state.env_cfg, 
                                                                            obs[i]['player_0'], 
                                                                            0), 
                             range(first_step, len(json_obs))))
    valid_action_1 = list(map(lambda i: action_parser.get_valid_actions_from_json(proxy.state.env_cfg, 
                                                                            obs[i]['player_1'],
                                                                            1), 
                             range(first_step, len(json_obs))))
    valid_actions = dict(player_0=valid_action_0, player_1=valid_action_1)
    return obs_, actions, valid_actions

if __name__ == "__main__":
    obs_, actions, valid_actions = replay_to_state_action(path="../test.json", max_length=1000)
    from policy.net import Net
    import torch
    import tree
    env = LuxEnv()
    mb_obs = env.concatenate_obs(obs_['player_0'][:4])
    mb_va = env.concatenate_va(valid_actions['player_0'][:4])
    mb_actions = env.concatenate_action(actions['player_0'][:4])

    agent = Net()
    np2torch = lambda x, dtype: torch.tensor(x).type(dtype)
    agent.bc_new(
                    np2torch(mb_obs['global_feature'], torch.float32),
                    np2torch(mb_obs['map_feature'], torch.float32), 
                    tree.map_structure(lambda x: np2torch(x, torch.int16), mb_obs['action_feature']),
                    tree.map_structure(lambda x: np2torch(x, torch.bool), mb_va),
                    tree.map_structure(lambda x: np2torch(x, torch.float32), mb_actions),
                )
    