import numpy as np
from .dense_reward_parser import DenseRewardParser
from impl_config import RewardParam


class SparseRewardParser(DenseRewardParser):

    def parse(self, dones, game_state, env_stats, global_info):
        _, sub_rewards = super(SparseRewardParser, self).parse(dones, game_state, env_stats, global_info)

        factory_count = [global_info[f'player_{pid}']["factory_count"] for pid in range(2)]
        lichen_count = [global_info[f'player_{pid}']["lichen_count"] for pid in range(2)]
        if dones["player_0"] and dones["player_1"]:
            reward = self.get_policy_score(
                factory_count[0],
                factory_count[1],
                lichen_count[0],
                lichen_count[1],
            )
        else:
            reward = [0., 0.]

        return reward, sub_rewards

    @staticmethod
    def get_policy_score(factory0_count, factory1_count, lichen0_count, lichen1_count):
        if factory0_count == 0 and factory1_count == 0:
            score = [0.5, 0.5]
        elif factory0_count != 0 and factory1_count != 0:
            if lichen0_count > lichen1_count:
                score = [1., 0.]
            elif lichen0_count < lichen1_count:
                score = [0., 1.]
            else:
                score = [0.5, 0.5]
        else:
            score = [
                float(factory0_count > 0),
                float(factory1_count > 0),
            ]
        return score