import os
import pickle
import argparse
from policy.net import Net
from luxenv import LuxEnv
import tree
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from impl_config import ModelParam, ActDims
import time
from luxs.load_from_replay import replay_to_state_action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda",
        help="device to be used ('cuda' or 'cpu')")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="learning rate for the optimizer")
    parser.add_argument("--minibatch-size", type=int, default=32,
        help="size of the minibatches")
    parser.add_argument("--replay-dir", type=str, default="./replays/replay_pool",
        help="directory of the replay files")
    parser.add_argument("--max-epochs", type=int, default=100,
        help="maximum number of epochs for training")
    parser.add_argument("--save-interval", type=int, default=100,
        help="epochs of saving model")   
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()

    start_time = time.time()
    device = args.device
    learning_rate = args.learning_rate
    minibatch_size = args.minibatch_size
    replay_dir = args.replay_dir

    env = LuxEnv()
    env.reset()
    agent = Net().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    np2torch = lambda x, dtype: torch.tensor(x).type(dtype).to(device)
    count = 0
    max_epochs = args.max_epochs
    for _ in range(max_epochs):
        for filename in os.listdir(replay_dir):
            if filename.endswith('.json.gz'):
                file_path = os.path.join(replay_dir, filename)
                b_obs,b_actions, b_va = replay_to_state_action(path=file_path)
                for player_id, player in enumerate(['player_0','player_1']):
                    b_inds = np.arange(len(b_obs[player]))
                    for start in range(0, len(b_obs[player]), minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        mb_obs = env.concatenate_obs(list(map(lambda i: b_obs[player][i], mb_inds)))
                        mb_va = env.concatenate_va(list(map(lambda i: b_va[player][i], mb_inds)))
                        mb_actions = env.concatenate_action(list(map(lambda i: b_actions[player][i], mb_inds)))
                        # behavior cloning
                        loss = agent.bc(
                                    np2torch(mb_obs['global_feature'], torch.float32),
                                    np2torch(mb_obs['map_feature'], torch.float32), 
                                    tree.map_structure(lambda x: np2torch(x, torch.int16), mb_obs['action_feature']),
                                    tree.map_structure(lambda x: np2torch(x, torch.bool), mb_va),
                                    tree.map_structure(lambda x: np2torch(x, torch.float32), mb_actions),
                                )
                        if loss:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    count +=1
                    if count>=args.save_interval:
                        torch.save(agent.state_dict(), "./model.pth")
                        print(f"save successfully! time cost:{time.time()-start_time}")
                        count = 0
