import numpy as np
from copy import deepcopy
from impl_config import RewardParam
from scipy.stats import gamma
from luxs.kit import EnvConfig


class GammaTransform:
    alpha: int = 2
    beta: float = 250
    gamma_rv = gamma(alpha, scale=beta)  # gamma distribution
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


class DenseRewardParser:

    def __init__(self, ):
        pass

    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def parse(self, dones, game_state, env_stats, global_info):
        sub_rewards_keys = [
            "reward_light",
            "reward_heavy",
            "reward_ice",
            "reward_ore",
            "reward_water",
            "reward_metal",
            "reward_lichen",
            "reward_factory",
            "reward_survival",
            "reward_win_lose",
        ]
        sub_rewards = [
            {k: 0
             for k in sub_rewards_keys},
            {k: 0
             for k in sub_rewards_keys},
        ]
        if RewardParam.use_gamma_coe:
            gamma_coe = GammaTransform.gamma_(game_state[0].real_env_steps)
            gamma_flipped_coe = GammaTransform.gamma_flipped(game_state[0].real_env_steps)
        else:
            gamma_coe, gamma_flipped_coe = 1, 1
        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            enm_global_info = global_info[f"player_{1 - team}"]
            last_count = self.last_count[player]
            own_sub_rewards = sub_rewards[team]

            factories_increment = own_global_info["factory_count"] - last_count['factory_count']
            light_increment = own_global_info["light_count"] - last_count['light_count']
            heavy_increment = own_global_info["heavy_count"] - last_count['heavy_count']
            ice_increment = own_global_info["total_ice"] - last_count['total_ice']
            ore_increment = own_global_info["total_ore"] - last_count['total_ore']
            water_increment = own_global_info["total_water"] - last_count['total_water']
            metal_increment = own_global_info["total_metal"] - last_count['total_metal']
            power_increment = own_global_info["total_power"] - last_count['total_power']
            lichen_increment = own_global_info["lichen_count"] - last_count['lichen_count']

            own_sub_rewards["reward_light"] = light_increment * RewardParam.light_reward_weight * gamma_flipped_coe
            own_sub_rewards["reward_heavy"] = heavy_increment * RewardParam.heavy_reward_weight
            own_sub_rewards["reward_ice"] = max(ice_increment, 0) * RewardParam.ice_reward_weight
            own_sub_rewards["reward_ore"] = max(ore_increment, 0) * RewardParam.ore_reward_weight * gamma_coe
            own_sub_rewards["reward_water"] = water_increment * RewardParam.water_reward_weight * gamma_flipped_coe
            own_sub_rewards["reward_metal"] = metal_increment * RewardParam.metal_reward_weight * gamma_coe
            own_sub_rewards["reward_power"] = power_increment * RewardParam.power_reward_weight
            own_sub_rewards["reward_lichen"] = lichen_increment * RewardParam.lichen_reward_weight * gamma_flipped_coe
            own_sub_rewards["reward_factory"] = factories_increment * RewardParam.factory_penalty_weight
            own_sub_rewards["reward_survival"] = RewardParam.survive_reward_weight * gamma_flipped_coe

            if dones[f'player_{team}']:
                own_lichen = own_global_info["lichen_count"]
                enm_lichen = enm_global_info["lichen_count"]
                if enm_global_info["factory_count"] == 0:
                    win = True
                elif own_lichen > enm_lichen:
                    win = True
                else:
                    win = False

                if win:
                    own_sub_rewards["reward_win_lose"] = RewardParam.win_reward_weight * (own_lichen - enm_lichen)**0.5
                else:
                    all_past_reward = 0
                    all_past_reward += own_global_info["light_count"] * RewardParam.light_reward_weight
                    all_past_reward += own_global_info["heavy_count"] * RewardParam.heavy_reward_weight
                    all_past_reward += own_global_info["total_ice"] * RewardParam.ice_reward_weight
                    all_past_reward += own_global_info["total_ore"] * RewardParam.ore_reward_weight
                    all_past_reward += own_global_info["total_water"] * RewardParam.water_reward_weight
                    all_past_reward += own_global_info["total_metal"] * RewardParam.metal_reward_weight
                    all_past_reward += own_global_info["total_power"] * RewardParam.power_reward_weight
                    all_past_reward += own_global_info["lichen_count"] * RewardParam.lichen_reward_weight
                    all_past_reward += own_global_info["factory_count"] * RewardParam.factory_penalty_weight
                    all_past_reward += game_state[team].env_steps * RewardParam.survive_reward_weight
                    own_sub_rewards["reward_win_lose"] = RewardParam.lose_penalty_coe * all_past_reward

        rewards = [sum(sub_rewards[0].values()), sum(sub_rewards[1].values())]

        # record total reward for logging
        # it is not added to the training rewards
        sub_rewards[0]["reward_total"] = sum(sub_rewards[0].values())
        sub_rewards[1]["reward_total"] = sum(sub_rewards[1].values())

        # ensure it is a zero-sum game
        if RewardParam.zero_sum:
            rewards_mean = sum(rewards) / 2
            rewards[0] -= rewards_mean
            rewards[1] -= rewards_mean

        self.update_last_count(global_info)

        return rewards, sub_rewards

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
