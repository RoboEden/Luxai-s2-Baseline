import argparse
import os
import random
import time
from distutils.util import strtobool
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from policy.net import Net
from luxenv import LuxSyncVectorEnv,LuxEnv
import tree
import json
import gzip
from luxs.load_from_replay import replay_to_state_action, get_obs_action_from_json
from utils import save_args, save_model, load_model, eval_model, _process_eval_resluts, cal_mean_return, make_env

LOG = True

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LuxAI_S2-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--train-num-collect", type=int, default=2048,
        help="the number of data collections in training process")
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-interval", type=int, default=100000, 
        help="global step interval to save model")
    parser.add_argument("--load-model-path", type=str, default=None,
        help="path for pretrained model loading")
    parser.add_argument("--evaluate-interval", type=int, default=10000,
        help="evaluation steps")
    parser.add_argument("--evaluate-num", type=int, default=5,
        help="evaluation numbers")
    parser.add_argument("--replay-dir", type=str, default=None,
        help="replay dirs to reset state")
    parser.add_argument("--eval", type=bool, default=False,
        help="is eval model")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    
    args.train_num_collect = args.minibatch_size if args.train_num_collect is None else args.train_num_collect
    args.minibatch_size = int(args.train_num_collect // args.num_minibatches)
    args.max_train_step = int(args.train_num_collect // args.num_envs)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

if __name__ == "__main__":
    args = parse_args()
    player_id = 0
    enemy_id = 1 - player_id
    player = f'player_{player_id}'
    enemy = f'player_{enemy_id}'
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_path = f'runs/{run_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if LOG:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    save_args(args, save_path+'args.json')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    envs = LuxSyncVectorEnv(
        [make_env(args.seed + i,args.replay_dir) for i in range(args.num_envs)]
    )
    

    agent = Net().to(device)
    if args.load_model_path is not None:
        agent.load_state_dict(torch.load(args.load_model_path))
        print('load successfully')
        if args.eval:
            import sys
            for i in range(10):
                eval_results = []
                for _ in range(args.evaluate_num):
                    eval_results.append(eval_model(agent))
                eval_results = _process_eval_resluts(eval_results)
                if LOG:
                    for key, value in eval_results.items():
                        writer.add_scalar(f"eval/{key}", value, i)
                pprint(eval_results)
            sys.exit()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup


    # start the game
    global_step = 0
    last_save_model_step = 0
    last_eval_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    for update in range(1, num_updates + 1):
        obs = dict(player_0=list(),player_1=list())
        actions = dict(player_0=list(),player_1=list())
        valid_actions = dict(player_0=list(),player_1=list())
        logprobs = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device), player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
        rewards = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
        dones = torch.zeros((args.max_train_step, args.num_envs)).to(device)
        values = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
        next_obs = envs.reset()
        next_done = torch.zeros(args.num_envs).to(device)

        total_done = 0
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        total_return = 0.0
        episode_return = torch.zeros(args.num_envs)
        episode_return_list = []
        episode_sub_return = {}
        episode_sub_return_list = []
        train_step = -1
        
        for step in range(0, args.num_steps):
            action = dict() 
            train_step += 1 
            global_step += 1 * args.num_envs
            for player_id, player in enumerate(['player_0', 'player_1']):
                obs[player] += envs.split(next_obs[player])
                dones[train_step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    valid_action = envs.get_valid_actions(player_id)
                    np2torch = lambda x, dtype: torch.tensor(x).type(dtype).to(device)
                    logprob, value, raw_action, _ = agent(
                        np2torch(next_obs[player]['global_feature'], torch.float32),
                        np2torch(next_obs[player]['map_feature'], torch.float32), 
                        tree.map_structure(lambda x: np2torch(x, torch.int16), next_obs[player]['action_feature']),
                        tree.map_structure(lambda x: np2torch(x, torch.bool), valid_action)
                    )
                    values[player][train_step] = value
                    valid_actions[player] += envs.split(valid_action)
                actions[player] += envs.split(raw_action)
                action[player_id] = raw_action
                logprobs[player][train_step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.
            async_action = envs.split(action)
            next_obs, reward, done, info = envs.step(async_action)
            for player_id, player in enumerate(['player_0', 'player_1']):
                rewards[player][train_step] = torch.tensor(reward[:, player_id]).to(device).view(-1)
            next_done = torch.tensor(done, dtype=torch.long).to(device)

            episode_return += torch.mean(torch.tensor(reward), dim=-1).to(episode_return.device)
            for key in info[0]['agents'][0]['sub_rewards']:
                if key not in episode_sub_return:
                    episode_sub_return[key] = torch.zeros(args.num_envs)
                sub_reward = list(map(lambda e_info: np.mean([e_info['agents'][0]['sub_rewards'][key], e_info['agents'][1]['sub_rewards'][key]]), info))
                episode_sub_return[key] += torch.tensor(np.array(sub_reward)).to(episode_sub_return[key].device)
            
            if True in next_done:
                episode_return_list.append(np.mean(episode_return[torch.where(next_done.cpu()==True)].cpu().numpy()))
                episode_return[torch.where(next_done==True)] = 0
                tmp_sub_return_dict = {}
                for key in episode_sub_return:
                    tmp_sub_return_dict.update({key: np.mean(episode_sub_return[key][torch.where(next_done.cpu()==True)].cpu().numpy())})
                    episode_sub_return[key][torch.where(next_done.cpu()==True)] = 0
                episode_sub_return_list.append(tmp_sub_return_dict)

            total_return += cal_mean_return(info, player_id=0)
            total_return += cal_mean_return(info, player_id=1)
            if (step== args.num_steps-1):
                print(f"global_step={global_step}, total_return={np.mean(episode_return_list)}")
                if LOG:
                    writer.add_scalar("charts/avg_steps", (step*args.num_envs)/total_done, global_step)
                    writer.add_scalar("charts/episodic_total_return", np.mean(episode_return_list), global_step)
                    mean_episode_sub_return = {}
                    for key in episode_sub_return.keys():
                        mean_episode_sub_return[key] = np.mean(list(map(lambda sub: sub[key], episode_sub_return_list)))
                        writer.add_scalar(f"sub_reward/{key}", mean_episode_sub_return[key], global_step)
                break
        # bootstrap value if not done
            if train_step >= args.max_train_step-1 or step == args.num_steps-1:  
                print("training ")
                returns = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                with torch.no_grad():
                    for player_id, player in enumerate(['player_0', 'player_1']):
                        _, next_value, _, _ = agent(
                                np2torch(next_obs[player]['global_feature'], torch.float32),
                                np2torch(next_obs[player]['map_feature'], torch.float32), 
                                tree.map_structure(lambda x: np2torch(x, torch.int16), next_obs[player]['action_feature']),
                                tree.map_structure(lambda x: np2torch(x, torch.bool), valid_action)
                            )
                        next_value = next_value.reshape(1,-1)
                        advantages = torch.zeros((args.max_train_step, args.num_envs)).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.max_train_step-1)):
                            if t == args.max_train_step - 1:
                                nextnonterminal = 1.0 - next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                nextvalues = values[player][t + 1]
                            delta = rewards[player][t] + args.gamma * nextvalues * nextnonterminal - values[player][t]
                            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        returns[player] = advantages + values[player]

                # flatten the batch
                b_obs = obs   
                b_logprobs = tree.map_structure(lambda x: x.reshape(-1), logprobs)
                b_actions = actions
                b_advantages = advantages.reshape(-1)
                b_returns = tree.map_structure(lambda x: x.reshape(-1), returns)
                b_values = tree.map_structure(lambda x: x.reshape(-1), values)
                b_va = valid_actions

                # Optimizing the policy and value network
                b_inds = np.arange(args.train_num_collect)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for player_id, player in enumerate(['player_0', 'player_1']):
                        for start in range(0, args.train_num_collect, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]
                            mb_obs = envs.concatenate_obs(list(map(lambda i: b_obs[player][i], mb_inds)))
                            mb_va = envs.concatenate_va(list(map(lambda i: b_va[player][i], mb_inds)))
                            mb_actions = envs.concatenate_action(list(map(lambda i: b_actions[player][i], mb_inds)))
                            newlogprob, newvalue, _, entropy = agent(
                                np2torch(mb_obs['global_feature'], torch.float32),
                                np2torch(mb_obs['map_feature'], torch.float32), 
                                tree.map_structure(lambda x: np2torch(x, torch.int16), mb_obs['action_feature']),
                                tree.map_structure(lambda x: np2torch(x, torch.bool), mb_va), 
                                tree.map_structure(lambda x: np2torch(x, torch.float32), mb_actions)
                            )
                            logratio = newlogprob - b_logprobs[player][mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                if len(mb_inds)==1:
                                    mb_advantages = mb_advantages
                                else:
                                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            newvalue = newvalue.view(-1)
                            if args.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[player][mb_inds]) ** 2
                                v_clipped = b_values[player][mb_inds] + torch.clamp(
                                    newvalue - b_values[player][mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[player][mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[player][mb_inds]) ** 2).mean()
                            entropy_loss = entropy.mean()
                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                            optimizer.step()

                        if args.target_kl is not None:
                            if approx_kl > args.target_kl:
                                break
                        y_pred, y_true = b_values[player].cpu().numpy(), b_returns[player].cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                        # TRY NOT TO MODIFY: record rewards for plotting purposes
                        if LOG:
                            writer.add_scalar(f"losses/value_loss_{player_id}", v_loss.item(), global_step)
                            writer.add_scalar(f"losses/policy_loss_{player_id}", pg_loss.item(), global_step)
                            writer.add_scalar(f"losses/entropy_{player_id}", entropy_loss.item(), global_step)
                            writer.add_scalar(f"losses/old_approx_kl_{player_id}", old_approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/approx_kl_{player_id}", approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/clipfrac_{player_id}", np.mean(clipfracs), global_step)
                            writer.add_scalar(f"losses/explained_variance_{player_id}", explained_var, global_step)
                
                if LOG:
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                print("global step: ", global_step)
                total_done += dones.sum()
                obs = dict(player_0=list(),player_1=list())
                actions = dict(player_0=list(),player_1=list())
                valid_actions = dict(player_0=list(),player_1=list())
                logprobs = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device), player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                rewards = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                dones = torch.zeros((args.max_train_step, args.num_envs)).to(device)
                values = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                train_step = -1
            # eval model
            if (global_step - last_eval_step) >= args.evaluate_interval:
                eval_results = []
                for _ in range(args.evaluate_num):
                    eval_results.append(eval_model(agent))
                eval_results = _process_eval_resluts(eval_results)
                if LOG:
                    for key, value in eval_results.items():
                        writer.add_scalar(f"eval/{key}", value, global_step)
                pprint(eval_results)
                last_eval_step = global_step
            # save model
            if (global_step - last_save_model_step) >= args.save_interval:
                save_model(agent, save_path+f'model_{global_step}.pth')
                last_save_model_step = global_step
            

    envs.close()
    if LOG:
        writer.close()
    
    
    
    