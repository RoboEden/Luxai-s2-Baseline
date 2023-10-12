from typing import Dict, List
import numpy as np
from copy import deepcopy

import tree
from impl_config import FactoryActType, RewardParam2, stats_reward_params
from scipy.stats import gamma
from luxs.kit import EnvConfig
from functools import reduce


class GammaTransform:
    alpha: int = 2
    beta: float = 250
    gamma_rv = gamma(alpha, scale=beta) 
    range_start = 50
    range_end = range_start + EnvConfig.max_episode_length
    scale = EnvConfig.max_episode_length * (1 / (gamma_rv.cdf(range_end) - gamma_rv.cdf(range_start)))

    @classmethod
    def gamma_(cls, x):
        y = cls.gamma_rv.pdf(x + cls.range_start) * cls.scale
        return y

    @classmethod
    def gamma_flipped(cls, x):
        y = cls.gamma_rv.pdf(cls.range_end - (x + cls.range_start)) * cls.scale
        return y


class Dense2RewardParser:

    def __init__(self, ):
        self.last_env_stats = None

    def reset(self, game_state, global_info, env_stats):
        self.update_env_stats(env_stats)

    def parse(self, dones, game_state, env_stats, global_info):
        sub_rewards = {0: {}, 1: {}}
        env_stats_diff = {}
        env_stats_rewards = {}
        for player in ["player_0", "player_1"]:
            env_stats_diff[player] = tree.map_structure(
                lambda cur, last: cur - last,
                env_stats[player],
                self.last_env_stats[player],
            )
            env_stats_rewards[player] = tree.map_structure(
                lambda x, param: x * param,
                env_stats_diff[player],
                stats_reward_params,
            )

        if RewardParam2.use_gamma_coe:
            gamma_coe = GammaTransform.gamma_(game_state[0].real_env_steps)
            gamma_flipped_coe = GammaTransform.gamma_flipped(game_state[0].real_env_steps)
        else:
            gamma_coe, gamma_flipped_coe = 1, 1

        for team in [0, 1]:
            player = f"player_{team}"
            opp_team = 1 - team
            opp_player = f"player_{opp_team}"
            if RewardParam2.balance_reward_punishment:
                env_stats_rewards[player]["generation"]["build"]["LIGHT"] = self.unit_generation_destroyed(
                    global_info["light_count"][team],
                    "build",
                    RewardParam2.light_min,
                ) * env_stats_rewards[player]["generation"]["build"]["LIGHT"]
                env_stats_rewards[player]["generation"]["build"]["HEAVY"] = self.unit_generation_destroyed(
                    global_info["heavy_count"][team],
                    "build",
                    RewardParam2.heavy_min,
                ) * env_stats_rewards[player]["generation"]["build"]["HEAVY"]

                env_stats_rewards[player]["destroyed"]["LIGHT"] = self.unit_generation_destroyed(
                    global_info["light_count"][team],
                    "destroyed",
                    RewardParam2.light_min,
                ) * env_stats_rewards[player]["destroyed"]["LIGHT"]
                env_stats_rewards[player]["destroyed"]["HEAVY"] = self.unit_generation_destroyed(
                    global_info["heavy_count"][team],
                    "destroyed",
                    RewardParam2.heavy_min,
                ) * env_stats_rewards[player]["destroyed"]["HEAVY"]
            else:
                pass
            if RewardParam2.use_gamma_coe:
                env_stats_rewards[player]["generation"]["lichen"] *= gamma_flipped_coe
                for cons in ["water", "metal", "ore", "ice"]:
                    if isinstance(env_stats_rewards[player]["generation"][cons], dict):
                        env_stats_rewards[player]["generation"][cons]["LIGHT"] *= gamma_flipped_coe
                        env_stats_rewards[player]["generation"][cons]["HEAVY"] *= gamma_flipped_coe
                    else:
                        env_stats_rewards[player]["generation"][cons] *= gamma_flipped_coe
            sub_rewards[team]["reward_survival"] = RewardParam2.survive_reward_weight * gamma_flipped_coe

            if RewardParam2.beat_opponent:
                sub_rewards[team]["reward_factories_opp_destroyed"] = env_stats_diff[opp_player]["destroyed"][
                    "FACTORY"] * RewardParam2.factories_opp_destroyed
                sub_rewards[team]["reward_unit_light_opp_destroyed"] = env_stats_diff[opp_player]["destroyed"][
                    "LIGHT"] * RewardParam2.unit_light_opp_destroyed
                sub_rewards[team]["reward_unit_heavy_opp_destroyed"] = env_stats_diff[opp_player]["destroyed"][
                    "HEAVY"] * RewardParam2.unit_heavy_opp_destroyed
                sub_rewards[team]["reward_lichen_opp_destroyed"] = sum(
                    env_stats_diff[opp_player]["destroyed"]["lichen"].values()) * RewardParam2.lichen_opp_destroyed
            else:
                sub_rewards[team]["reward_factories_opp_destroyed"] = 0
                sub_rewards[team]["reward_unit_light_opp_destroyed"] = 0
                sub_rewards[team]["reward_unit_heavy_opp_destroyed"] = 0
                sub_rewards[team]["reward_lichen_opp_destroyed"] = 0

            if dones[f'player_{team}']:
                if global_info[opp_player]["factory_count"] == 0:
                    win = True
                elif global_info[player]["lichen_count"] > global_info[opp_player]["lichen_count"]:
                    win = True
                else:
                    win = False

                if win:  # win by lichen
                    sub_rewards[team]["reward_win_lose"] = RewardParam2.win_reward_weight
                else:
                    sub_rewards[team]["reward_win_lose"] = 0
            else:
                sub_rewards[team]["reward_win_lose"] = 0
        for team in [0, 1]:
            player = f"player_{team}"
            env_stats_rewards[player] = tree.flatten_with_path(env_stats_rewards[player])
            env_stats_rewards[player] = list(
                map(
                    lambda item: {"reward_" + "_".join(item[0]).lower(): item[1]},
                    env_stats_rewards[player],
                ))
            env_stats_rewards[player] = reduce(lambda cat1, cat2: dict(cat1, **cat2), env_stats_rewards[player])
            sub_rewards[team].update(env_stats_rewards[player])

        rewards = [sum(sub_rewards[0].values()), sum(sub_rewards[1].values())]

        # record total reward for logging
        # it is not added to the training rewards
        sub_rewards[0]["reward_total"] = sum(sub_rewards[0].values())
        sub_rewards[1]["reward_total"] = sum(sub_rewards[1].values())

        # ensure it is a zero-sum game
        if RewardParam2.zero_sum:
            rewards_mean = sum(rewards) / 2
            rewards[0] -= rewards_mean
            rewards[1] -= rewards_mean

        self.update_env_stats(env_stats)
        return rewards, sub_rewards

    def update_env_stats(self, env_stats: dict):
        self.last_env_stats = deepcopy(env_stats)

    @staticmethod
    def unit_generation_destroyed(num_robot: int, mode: str, min_point=10):
        minima = 2 * (min_point)
        func = lambda num_robot: (num_robot + min_point**2 / num_robot) / minima
        if num_robot == 0:
            ratio = func(1) * 2
        elif num_robot <= min_point:
            ratio = func(num_robot)
        elif num_robot > min_point:
            ratio = 1 / func(num_robot)

        if mode == "build":
            return ratio
        elif mode == "destroyed":
            return 1 / ratio
        else:
            return None