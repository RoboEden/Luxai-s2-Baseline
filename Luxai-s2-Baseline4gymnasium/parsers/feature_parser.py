import numpy as np
import tree

from impl_config import EnvParam
import luxs.kit
from typing import NamedTuple
from luxai_s2.config import EnvConfig
from functools import reduce


class LuxFeature(NamedTuple):
    global_feature: np.ndarray
    map_feature: np.ndarray
    action_feature: dict


class FeatureParser():

    def __init__(self):

        self.global_feature_names = [
            'env_step',
            'cycle',
            'hour',
            'daytime_or_night',
            'num_factory_own',
            'num_factory_enm',
            'total_lichen_own',
            'total_lichen_enm',
        ]
        for own_enm in ['own', 'enm']:
            self.global_feature_names += [
                f'factory_total_power_{own_enm}',
                f'factory_total_ice_{own_enm}',
                f'factory_total_water_{own_enm}',
                f'factory_total_ore_{own_enm}',
                f'factory_total_metal_{own_enm}',
                f'num_light_{own_enm}',
                f'num_heavy_{own_enm}',
                f'robot_total_power_{own_enm}',
                f'robot_total_ice_{own_enm}',
                f'robot_total_water_{own_enm}',
                f'robot_total_ore_{own_enm}',
                f'robot_total_metal_{own_enm}',
            ]

        self.map_featrue_names = [
            'ice',
            'ore',
            'rubble',
            'lichen',
            'lichen_strains',
            'lichen_strains_own',
            'lichen_strains_enm',
            'valid_region_indicator',
            'factory_id',
            'factory_power',
            'factory_ice',
            'factory_water',
            'factory_ore',
            'factory_metal',
            'factory_own',
            'factory_enm',
            'factory_can_build_light',
            'factory_can_build_heavy',
            'factory_can_grow_lichen',
            'factory_water_cost',
            'unit_id',
            'unit_power',
            'unit_ice',
            'unit_water',
            'unit_ore',
            'unit_metal',
            'unit_own',
            'unit_enm',
            'unit_light',
            'unit_heavy',
        ]

        self.global_info_names = [
            'factory_count',
            'light_count',
            'heavy_count',
            'unit_ice',
            'unit_ore',
            'unit_water',
            'unit_metal',
            'unit_power',
            'factory_ice',
            'factory_ore',
            'factory_water',
            'factory_metal',
            'factory_power',
            'total_ice',
            'total_ore',
            'total_water',
            'total_metal',
            'total_power',
            'lichen_count',
        ]

    def parse(self, obs, env_cfg):
        all_feature = {}
        for player, player_obs in obs.items():
            env_step = player_obs['real_env_steps'] + player_obs['board']['factories_per_team'] * 2 + 1
            game_state = luxs.kit.obs_to_game_state(env_step, env_cfg, player_obs)
            parsed_feature = self._get_feature(obs=game_state, player=player)
            # all_feature.append(parsed_feature)
            all_feature[player] = parsed_feature
        global_info = {player: self._get_info(player, game_state) for player in ['player_0', 'player_1']}
        return all_feature, global_info
    
    def json_parser(self, obs, env_cfg):
        all_feature = {}
        for player, player_obs in obs.items():
            env_step = player_obs['real_env_steps'] + player_obs['board']['factories_per_team'] * 2 + 1
            game_state = luxs.kit.json_obs_to_game_state(env_step, env_cfg, player_obs)
            parsed_feature = self._get_feature(obs=game_state, player=player)
            # all_feature.append(parsed_feature)
            all_feature[player] = parsed_feature
        return all_feature

    def parse2(self, game_state, player):
        return self._get_feature(obs=game_state, player=player)

    def _get_info(self, player: str, obs: luxs.kit.GameState):
        global_info = {k: 0 for k in self.global_info_names}
        factories = list(obs.factories[player].values())
        units = list(obs.units[player].values())

        global_info['light_count'] = sum(int(u.unit_type == 'LIGHT') for u in units)
        global_info['heavy_count'] = sum(int(u.unit_type == 'HEAVY') for u in units)
        global_info["factory_count"] = len(factories)

        global_info['unit_ice'] = sum(u.cargo.ice for u in units)
        global_info['unit_ore'] = sum(u.cargo.ore for u in units)
        global_info['unit_water'] = sum(u.cargo.water for u in units)
        global_info['unit_metal'] = sum(u.cargo.metal for u in units)
        global_info['unit_power'] = sum(u.power for u in units)

        global_info['factory_ice'] = sum(f.cargo.ice for f in factories)
        global_info['factory_ore'] = sum(f.cargo.ore for f in factories)
        global_info['factory_water'] = sum(f.cargo.water for f in factories)
        global_info['factory_metal'] = sum(f.cargo.metal for f in factories)
        global_info['factory_power'] = sum(f.power for f in factories)

        global_info['total_ice'] = global_info['unit_ice'] + global_info['factory_ice']
        global_info['total_ore'] = global_info['unit_ore'] + global_info['factory_ore']
        global_info['total_water'] = global_info['unit_water'] + global_info['factory_water']
        global_info['total_metal'] = global_info['unit_metal'] + global_info['factory_metal']
        global_info['total_power'] = global_info['unit_power'] + global_info['factory_power']

        lichen = obs.board.lichen
        lichen_strains = obs.board.lichen_strains
        if factories:
            lichen_count = sum((np.sum(lichen[lichen_strains == f.strain_id]) for f in factories), 0)
            global_info['lichen_count'] = lichen_count
        else:
            global_info['lichen_count'] = 0
        return global_info

    def _get_feature(self, obs: luxs.kit.GameState, player: str, output_dict=True):
        env_cfg: EnvConfig = obs.env_cfg
        enemy = 'player_1' if player == 'player_0' else 'player_0'

        map_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.map_featrue_names}
        map_feature['ice'] = obs.board.ice
        map_feature['ore'] = obs.board.ore
        map_feature['rubble'] = obs.board.rubble
        map_feature['lichen'] = obs.board.lichen
        map_feature['lichen_strains'] = obs.board.lichen_strains
        map_feature['lichen_strains_own'] = sum(
            (obs.board.lichen_strains == f.strain_id for f in obs.factories[player].values()) if obs.factories else [],
            np.zeros_like(obs.board.lichen_strains, dtype=np.bool8),
        )
        map_feature['lichen_strains_enm'] = sum(
            (obs.board.lichen_strains == f.strain_id for f in obs.factories[enemy].values()) if obs.factories else [],
            np.zeros_like(obs.board.lichen_strains, dtype=np.bool8),
        )
        map_feature['valid_region_indicator'] = np.ones_like(obs.board.rubble)

        global_feature = {name: 0 for name in self.global_feature_names}
        global_feature['env_step'] = obs.real_env_steps
        global_feature['cycle'] = obs.real_env_steps // env_cfg.CYCLE_LENGTH
        global_feature['hour'] = obs.real_env_steps % env_cfg.CYCLE_LENGTH
        global_feature['daytime_or_night'] = global_feature['hour'] < 30
        global_feature['num_factory_own'] = len(obs.factories[player])
        global_feature['num_factory_enm'] = len(obs.factories[enemy])
        global_feature['total_lichen_own'] = np.sum(obs.board.lichen[map_feature['lichen_strains_own']])
        global_feature['total_lichen_enm'] = np.sum(obs.board.lichen[map_feature['lichen_strains_enm']])

        for own_enm, pid in zip(['own', 'enm'], [player, enemy]):
            global_feature[f'factory_total_power_{own_enm}'] = sum(f.power for f in obs.factories[pid].values())
            global_feature[f'factory_total_ice_{own_enm}'] = sum(f.cargo.ice for f in obs.factories[pid].values())
            global_feature[f'factory_total_water_{own_enm}'] = sum(f.cargo.water for f in obs.factories[pid].values())
            global_feature[f'factory_total_ore_{own_enm}'] = sum(f.cargo.ore for f in obs.factories[pid].values())
            global_feature[f'factory_total_metal_{own_enm}'] = sum(f.cargo.metal for f in obs.factories[pid].values())

            global_feature[f'num_light_{own_enm}'] = sum(u.unit_type == "LIGHT" for u in obs.units[pid].values())
            global_feature[f'num_heavy_{own_enm}'] = sum(u.unit_type == "HEAVY" for u in obs.units[pid].values())
            assert global_feature[f'num_light_{own_enm}'] + global_feature[f'num_heavy_{own_enm}'] == len(
                obs.units[pid])
            global_feature[f'robot_total_power_{own_enm}'] = sum(u.power for u in obs.units[pid].values())
            global_feature[f'robot_total_ice_{own_enm}'] = sum(u.cargo.ice for u in obs.units[pid].values())
            global_feature[f'robot_total_water_{own_enm}'] = sum(u.cargo.water for u in obs.units[pid].values())
            global_feature[f'robot_total_ore_{own_enm}'] = sum(u.cargo.ore for u in obs.units[pid].values())
            global_feature[f'robot_total_metal_{own_enm}'] = sum(u.cargo.metal for u in obs.units[pid].values())

        for owner, factories in obs.factories.items():
            for fid, factory in factories.items():
                x, y = factory.pos
                map_feature['factory_id'][x, y] = int(fid[len('factory_'):])
                map_feature['factory_power'][x, y] = factory.power
                map_feature['factory_ice'][x, y] = factory.cargo.ice
                map_feature['factory_water'][x, y] = factory.cargo.water
                map_feature['factory_ore'][x, y] = factory.cargo.ore
                map_feature['factory_metal'][x, y] = factory.cargo.metal
                map_feature['factory_own'][x, y] = owner == player
                map_feature['factory_enm'][x, y] = owner == enemy

                if (factory.cargo.metal >= env_cfg.ROBOTS['LIGHT'].METAL_COST) \
                    and (factory.power >= env_cfg.ROBOTS['LIGHT'].POWER_COST):
                    map_feature['factory_can_build_light'][x, y] = True
                if (factory.cargo.metal >= env_cfg.ROBOTS['HEAVY'].METAL_COST) \
                    and (factory.power >=env_cfg.ROBOTS['HEAVY'].POWER_COST):
                    map_feature['factory_can_build_heavy'][x, y] = True
                water_cost = np.sum(
                    obs.board.lichen_strains == factory.strain_id) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1
                if factory.cargo.water >= water_cost:
                    map_feature['factory_can_grow_lichen'][x, y] = True
                map_feature['factory_water_cost'][x, y] = water_cost

        for owner, units in obs.units.items():
            for uid, unit in units.items():
                x, y = unit.pos
                map_feature['unit_id'][x, y] = int(uid[len('unit_'):])
                map_feature['unit_power'][x, y] = unit.power
                map_feature['unit_ice'][x, y] = unit.cargo.ice
                map_feature['unit_water'][x, y] = unit.cargo.water
                map_feature['unit_ore'][x, y] = unit.cargo.ore
                map_feature['unit_metal'][x, y] = unit.cargo.metal

                map_feature['unit_own'][x, y] = owner == player
                map_feature['unit_enm'][x, y] = owner == enemy

                map_feature['unit_light'][x, y] = unit.unit_type == "LIGHT"
                map_feature['unit_heavy'][x, y] = unit.unit_type == "HEAVY"
                assert unit.unit_type in ["LIGHT", "HEAVY"]

        # action queue
        action_feature = dict(
            unit_indicator=np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.int16),
            type=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            direction=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            resource=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            amount=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            repeat=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            n=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
        )
        empty_action = [0] * 6
        for units in obs.units.values():
            for unit in units.values():
                padding = [empty_action] * (env_cfg.UNIT_ACTION_QUEUE_SIZE - len(unit.action_queue))
                actions = np.array(list(unit.action_queue) + padding)

                x, y = unit.pos
                action_feature['unit_indicator'][x, y] = True
                action_feature['type'][x, y, :] = actions[:, 0]
                action_feature['direction'][x, y, :] = actions[:, 1]
                action_feature['resource'][x, y, :] = actions[:, 2]
                action_feature['amount'][x, y, :] = actions[:, 3]
                action_feature['repeat'][x, y, :] = actions[:, 4]
                action_feature['n'][x, y, :] = actions[:, 5]

        # normalize
        light_cfg = env_cfg.ROBOTS['LIGHT']
        map_feature

        map_feature['rubble'] = map_feature['rubble'] / env_cfg.MAX_RUBBLE
        map_feature['lichen'] = map_feature['lichen'] / env_cfg.MAX_LICHEN_PER_TILE

        map_feature['factory_power'] = map_feature['factory_power'] / light_cfg.BATTERY_CAPACITY
        map_feature['unit_power'] = map_feature['unit_power'] / light_cfg.BATTERY_CAPACITY

        map_feature['factory_ice'] = map_feature['factory_ice'] / light_cfg.CARGO_SPACE
        map_feature['factory_water'] = map_feature['factory_water'] / light_cfg.CARGO_SPACE
        map_feature['factory_ore'] = map_feature['factory_ore'] / light_cfg.CARGO_SPACE
        map_feature['factory_metal'] = map_feature['factory_metal'] / light_cfg.CARGO_SPACE
        map_feature['factory_water_cost'] = map_feature['factory_water_cost'] / light_cfg.CARGO_SPACE
        map_feature['unit_ice'] = map_feature['unit_ice'] / light_cfg.CARGO_SPACE
        map_feature['unit_water'] = map_feature['unit_water'] / light_cfg.CARGO_SPACE
        map_feature['unit_ore'] = map_feature['unit_ore'] / light_cfg.CARGO_SPACE
        map_feature['unit_metal'] = map_feature['unit_metal'] / light_cfg.CARGO_SPACE

        global_feature['total_lichen_own'] = global_feature['total_lichen_own'] / env_cfg.MAX_LICHEN_PER_TILE
        global_feature['total_lichen_enm'] = global_feature['total_lichen_enm'] / env_cfg.MAX_LICHEN_PER_TILE

        for own_enm in ['own', 'enm']:
            # yapf: disable
            global_feature[f'factory_total_power_{own_enm}'] = global_feature[f'factory_total_power_{own_enm}'] / light_cfg.BATTERY_CAPACITY
            global_feature[f'factory_total_ice_{own_enm}'] = global_feature[f'factory_total_ice_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_water_{own_enm}'] = global_feature[f'factory_total_water_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_ore_{own_enm}'] = global_feature[f'factory_total_ore_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_metal_{own_enm}'] = global_feature[f'factory_total_metal_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_power_{own_enm}'] = global_feature[f'robot_total_power_{own_enm}'] / light_cfg.BATTERY_CAPACITY
            global_feature[f'robot_total_ice_{own_enm}'] = global_feature[f'robot_total_ice_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_water_{own_enm}'] = global_feature[f'robot_total_water_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_ore_{own_enm}'] = global_feature[f'robot_total_ore_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_metal_{own_enm}'] = global_feature[f'robot_total_metal_{own_enm}'] / light_cfg.CARGO_SPACE
            # yapf: enable

        global_feature = np.array(list(global_feature.values()))
        map_feature = np.array(list(map_feature.values()))

        if output_dict:
            return {'global_feature': global_feature, 'map_feature': map_feature, 'action_feature': action_feature}

        return LuxFeature(global_feature, map_feature, action_feature)

    @staticmethod
    def log_env_stats(env_stats):
        for team in [0, 1]:
            player = f"player_{team}"
            stat = tree.flatten_with_path(env_stats[player])
            stat = list(map(
                lambda item: {"_".join(item[0]).lower(): item[1]},
                stat,
            ))
            stat = reduce(lambda cat1, cat2: dict(cat1, **cat2), stat)
            env_stats_logs = stat
        return env_stats_logs