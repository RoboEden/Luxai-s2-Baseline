import json
import sys
from argparse import Namespace
from typing import Dict
import torch.nn as nn
from policy.net import Net
from luxs.config import EnvConfig
from impl_config import ModelParam, ActDims
from luxs.kit import (
    GameState,
    from_json,
    obs_to_game_state,
    process_action,
    process_obs,
    to_json,
)
import tree
import contextlib
with contextlib.redirect_stdout(None):
    from parsers import ActionParser, FeatureParser
import torch
import numpy as np
from player import Player
### The model path
PATH = 'your/model/path.pth'
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    dict()
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()
    

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict, env_cfg
    step = observation.step
    player = observation.player
    player_id = 0 if player=='player_0' else 1
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Net()
        agent_prev_obs[player] = dict()
        agent = agent_dict[player]
        agent.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    game_state = obs_to_game_state(step, env_cfg, obs)
    agent_prev_obs[player] = obs
    agent.step = step


    def torch2np(x):
        if isinstance(x, torch.Tensor):
            return x[0].detach().cpu().numpy()
        else:
            return x
    if obs["real_env_steps"] < 0:
        action = Player(player,env_cfg).early_setup(step, obs)
    else:
        with torch.no_grad():
            obs = FeatureParser().parse2(game_state, player)
            valid_action = ActionParser().get_valid_actions(game_state, player_id)
            np2torch = lambda x, dtype: torch.tensor(x).unsqueeze(0).type(dtype)
            _,_,actions,_ = agent(torch.tensor(obs['global_feature'],dtype=torch.float).unsqueeze(0),
                                  torch.tensor(obs['map_feature'],dtype=torch.float).unsqueeze(0),\
                                tree.map_structure(lambda x: np2torch(x, torch.int16), obs['action_feature']),\
                                tree.map_structure(lambda x: np2torch(x, torch.bool), valid_action)
                                            )
            actions = tree.map_structure(lambda x: torch2np(x), actions)
            action = ActionParser().parse2(game_state, actions, player)
        
    return process_action(action)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    configurations = None
    i = 0
    
    while True:
        inputs = read_input()
        obs = json.loads(inputs)

        observation = Namespace(
            **dict(
                step=obs["step"],
                obs=json.dumps(obs["obs"]),
                remainingOverageTime=obs["remainingOverageTime"],
                player=obs["player"],
                info=obs["info"],
            )
        )
        if i==0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        # send actions to engine
        print(json.dumps(actions))
