from typing import Dict
import numpy as np
from impl_config import (
    ActDims,
    EnvParam,
    FactoryActType,
    ModelParam,
    ResourceType,
    UnitActType,
    UnitActChannel,
)
import luxs
from luxs.kit import GameState, json_obs_to_game_state
from luxs.utils import my_turn_to_place_factory
from luxai_s2 import actions as lux_actions
from luxai_s2.actions import move_deltas
import tree
import dataclasses

factory_adjacent_delta_xy = np.array([
    [-2, -1],
    [-2, +0],
    [-2, +1],
])
factory_adjacent_delta_xy = np.concatenate([factory_adjacent_delta_xy, -factory_adjacent_delta_xy])
factory_adjacent_delta_xy = np.concatenate([factory_adjacent_delta_xy, factory_adjacent_delta_xy[:, ::-1]])


def ind2vec(ind, shape):
    element_size = np.cumprod(shape[::-1])[::-1]
    element_size = np.concatenate([element_size, [1]])[1:].tolist()

    vec = []
    for ez in element_size:
        vec.append(ind // ez)
        ind %= ez
    return vec


def get_resource_amount(unit, resource):
    amount = [
        unit.cargo.ice,
        unit.cargo.ore,
        unit.cargo.water,
        unit.cargo.metal,
        unit.power,
    ][resource]
    return amount


class ActionParser():

    def __init__(self):
        self.agents = ['player_0', 'player_1']

    def _parse(self, game_state: GameState, player: str, raw_actions):
        player_id = int(player[-1])
        env_cfg = game_state.env_cfg
        if game_state.env_steps == 0:
            bid = raw_actions['bid'] - ActDims.bid // 2
            faction = "FirstMars"
            action = {
                'bid': bid,
                'faction': faction,
            }
            return action

        my_turn = my_turn_to_place_factory(game_state.teams[player].place_first, game_state.env_steps)
        if game_state.real_env_steps < 0:
            if my_turn:
                spawn = raw_actions['factory_spawn']
                # spawn = tree.map_structure(lambda x: x.cpu().detach().numpy(), spawn)
                x = int(spawn['location'] // EnvParam.map_size)
                y = int(spawn['location'] % EnvParam.map_size)
                if ModelParam.spawn_distribution in ['beta', 'normal']:
                    water_percent = spawn['water']
                    metal_percent = spawn['metal']
                elif ModelParam.spawn_distribution == 'categorical':
                    water_percent = (spawn['water'] + 1) / ActDims.amount
                    metal_percent = (spawn['metal'] + 1) / ActDims.amount
                elif ModelParam.spawn_distribution == 'script':
                    water_percent = env_cfg.INIT_WATER_METAL_PER_FACTORY / 200
                    metal_percent = env_cfg.INIT_WATER_METAL_PER_FACTORY / 200
                else:
                    raise NotImplementedError
                action = {
                    "spawn": [x, y],
                    "water": np.round(water_percent * 200),
                    "metal": np.round(metal_percent * 200),
                }
            else:
                action = {}
            return action

        player_actions = {}
        factories = game_state.factories[player]
        robots = game_state.units[player]

        ## factories' action: int
        for unit_id, factory in factories.items():
            x, y = factory.pos
            raw_action = raw_actions["factory_act"][x, y]
            if raw_action != FactoryActType.DO_NOTHING:
                player_actions[unit_id] = raw_action.item()

        ## robots' action: 5d array
        for unit_id, robot in robots.items():
            x, y = robot.pos
            action = raw_actions["unit_act"][:, x, y].copy()
            act_type = action[UnitActChannel.TYPE].astype(int)
            dir = action[UnitActChannel.DIRECTION].astype(int)
            resource = action[UnitActChannel.RESOURCE].astype(int)
            percentage = action[UnitActChannel.AMOUNT]
            if ModelParam.amount_distribution == "categorical":
                percentage = (percentage + 1) / ActDims.amount
            elif ModelParam.amount_distribution == "beta":
                pass
            else:
                raise NotImplementedError

            if act_type == UnitActType.MOVE:
                pass

            if act_type == UnitActType.TRANSFER:
                self_amount = get_resource_amount(robot, resource)
                target_pos = robot.pos + lux_actions.move_deltas[dir]

                # if the target is a unit
                for _, target in robots.items():
                    if np.array_equal(target.pos, target_pos):
                        target_amount = get_resource_amount(target, resource)
                        if resource == ResourceType.POWER:
                            space_limit = target.unit_cfg.BATTERY_CAPACITY
                        else:
                            space_limit = target.unit_cfg.CARGO_SPACE
                        target_space = space_limit - target_amount
                        break

                # if the target is a factory
                for _, target in factories.items():
                    if (np.abs(np.array(target_pos) - target.pos) <= 1).all():
                        target_space = env_cfg.max_transfer_amount
                        break

                action[UnitActChannel.AMOUNT] = round(min(target_space, self_amount) * percentage)

            if act_type == UnitActType.PICKUP:
                robot_amount = get_resource_amount(robot, resource)
                if resource == ResourceType.POWER:
                    space_limit = robot.unit_cfg.BATTERY_CAPACITY
                else:
                    space_limit = robot.unit_cfg.CARGO_SPACE
                robot_space = space_limit - robot_amount

                for _, target in factories.items():
                    if (np.abs(np.array(robot.pos) - target.pos) <= 1).all():
                        target_amount = get_resource_amount(target, resource)
                        break

                action[UnitActChannel.AMOUNT] = round(min(target_amount, robot_space) * percentage)

            if act_type == UnitActType.DIG:
                pass

            if act_type == UnitActType.SELF_DESTRUCT:
                pass

            if act_type == UnitActType.RECHARGE:
                action[UnitActChannel.AMOUNT] = robot.unit_cfg.BATTERY_CAPACITY

            if act_type == UnitActType.DO_NOTHING:
                continue

            player_actions[unit_id] = action.astype(np.int32)

        for unit_id, robot in robots.items():
            if unit_id in player_actions:
                # get current action
                if len(robot.action_queue) > 0:
                    current_action = robot.action_queue[0]
                else:
                    current_action = [0] * 6

                # only update queue if the current_action action is different from new action
                new_action = player_actions[unit_id]
                if not np.array_equal(current_action[:4], new_action[:4]):
                    player_actions[unit_id] = [new_action]
                else:
                    player_actions.pop(unit_id)
        return player_actions

    def inv_parse(self, obs, raw_actions, env_cfg):
        actions = {}
        for player in ['player_0', 'player_1']:
            player_id = int(player[-1])
            json_obs = obs[player]
            env_step = json_obs['real_env_steps'] + json_obs['board']['factories_per_team'] * 2 + 1
            game_state = luxs.kit.json_obs_to_game_state(env_step, env_cfg, json_obs)
            action = dict(factory_act=np.zeros((EnvParam.map_size,EnvParam.map_size)),
                        unit_act=np.zeros((len(UnitActChannel),EnvParam.map_size,EnvParam.map_size)))
            factories = game_state.factories[player]
            robots = game_state.units[player]

            ## factories' action: int
            for unit_id, factory in factories.items():
                x, y = factory.pos
                if unit_id in raw_actions[player_id].keys():
                    action["factory_act"][x, y] = raw_actions[player_id][unit_id]
                else:
                    action["factory_act"][x, y] = FactoryActType.DO_NOTHING

            ## robots' action: 5d array
            for unit_id, robot in robots.items():
                x, y = robot.pos
                if unit_id in raw_actions[player_id].keys():
                    if len(raw_actions[player_id][unit_id]) > 0:
                        raw_action = raw_actions[player_id][unit_id][0]
                        action["unit_act"][UnitActChannel.TYPE, x, y] = raw_action[UnitActChannel.TYPE]
                        action["unit_act"][UnitActChannel.DIRECTION, x, y] = raw_action[UnitActChannel.DIRECTION]
                        action["unit_act"][UnitActChannel.RESOURCE, x, y] = raw_action[UnitActChannel.RESOURCE]
                        action["unit_act"][UnitActChannel.AMOUNT, x, y] = raw_action[UnitActChannel.AMOUNT]
                        action["unit_act"][UnitActChannel.REPEAT, x, y] = raw_action[UnitActChannel.REPEAT]
                        action["unit_act"][UnitActChannel.N, x, y] = raw_action[UnitActChannel.N]
                    else:
                        action["unit_act"][UnitActChannel.TYPE, x, y] = UnitActType.DO_NOTHING
                else:
                    action["unit_act"][UnitActChannel.TYPE, x, y] = UnitActType.DO_NOTHING
            actions[player] = action
        return actions

    def parse(self, game_state, raw_actions):
        actions = {}
        for player_id in [0, 1]:
            player = self.agents[player_id]
            actions[player] = self._parse(game_state[player_id], player, raw_actions[player_id])

        action_stats = [
            self.action_stats(player, actions[player], game_state[i]) for i, player in enumerate(self.agents)
        ]

        return actions, action_stats

    def parse2(self, game_state, raw_actions, player):
        return self._parse(game_state, player, raw_actions)

    def get_valid_actions_from_json(self, env_cfg, obs, player_id):
        env_step = obs['real_env_steps'] + obs['board']['factories_per_team'] * 2 + 1
        game_state = json_obs_to_game_state(env_step, env_cfg, obs)
        return self.get_valid_actions(game_state, player_id)

    @staticmethod
    def get_valid_actions(game_state: GameState, player_id: int):
        player = 'player_0' if player_id == 0 else 'player_1'
        enemy = 'player_1' if player_id == 0 else 'player_0'
        board = game_state.board
        env_cfg = game_state.env_cfg

        # board = game_state.board

        def factory_under_unit(unit_pos, factories):
            for _, factory in factories.items():
                factory_pos = factory.pos
                if abs(unit_pos[0] - factory_pos[0]) <= 1 and abs(unit_pos[1] - factory_pos[1]) <= 1:
                    return factory
            return None

        act_dims_mapping = dataclasses.asdict(EnvParam.act_dims_mapping)

        valid_actions = tree.map_structure(
            lambda dim: np.zeros((dim, EnvParam.map_size, EnvParam.map_size), dtype=np.bool8), act_dims_mapping)

        factory_va = valid_actions["factory_act"]
        for unit_id, factory in game_state.factories[player].items():
            x, y = factory.pos
            # valid build light
            if factory.cargo.metal >= env_cfg.ROBOTS['LIGHT'].METAL_COST\
                and factory.power >= env_cfg.ROBOTS['LIGHT'].POWER_COST:
                factory_va[FactoryActType.BUILD_LIGHT, x, y] = True
            # valid build heavy
            if factory.cargo.metal >= env_cfg.ROBOTS['HEAVY'].METAL_COST\
                and factory.power >= env_cfg.ROBOTS['HEAVY'].POWER_COST:
                factory_va[FactoryActType.BUILD_HEAVY, x, y] = True
            # valid grow lichen
            lichen_strains_size = np.sum(board.lichen_strains == factory.strain_id)
            if factory.cargo.water >= (lichen_strains_size + 1) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1:
                adj_xy = factory.pos + factory_adjacent_delta_xy
                adj_xy = adj_xy[(adj_xy >= 0).all(axis=1) & (adj_xy < EnvParam.map_size).all(axis=1)]
                adj_x, adj_y = adj_xy[:, 0], adj_xy[:, 1]
                no_ruble = (board.rubble[adj_x, adj_y] == 0)
                no_ice = (board.ice[adj_x, adj_y] == 0)
                no_ore = (board.ore[adj_x, adj_y] == 0)
                if (no_ruble & no_ice & no_ore).any():
                    factory_va[FactoryActType.WATER, x, y] = True

            # always can do nothing
            factory_va[FactoryActType.DO_NOTHING, x, y] = True

        # construct unit_map
        unit_map = np.full_like(game_state.board.rubble, fill_value=-1, dtype=np.int32)
        for unit_id, unit in game_state.units[player].items():
            x, y = unit.pos
            unit_map[x, y] = int(unit_id[len("unit_"):])

        for unit_id, unit in game_state.units[player].items():
            x, y = unit.pos
            action_queue_cost = unit.action_queue_cost(game_state)
            if unit.power >= action_queue_cost:
                valid_actions["unit_act"]["act_type"][:, x, y] = True
            else:
                valid_actions["unit_act"]["act_type"][UnitActType.DO_NOTHING, x, y] = True
                continue

            # valid unit move
            valid_actions["unit_act"]["move"]["repeat"][:, x, y] = True
            for direction in range(len(move_deltas)):
                target_pos = unit.pos + move_deltas[direction]

                # always forbid to move to the same position
                if direction == 0:
                    continue

                if (target_pos[0] < 0 or target_pos[1] < 0 or target_pos[0] >= EnvParam.map_size
                        or target_pos[1] >= EnvParam.map_size):
                    continue

                if factory_under_unit(target_pos, game_state.factories[enemy]) is not None:
                    continue

                power_required = unit.move_cost(game_state, direction)
                if unit.power - action_queue_cost >= power_required:
                    valid_actions["unit_act"]["move"]["direction"][direction, x, y] = True

            # valid transfer
            valid_actions["unit_act"]["transfer"]['repeat'][0, x, y] = True
            amounts = [unit.cargo.ice, unit.cargo.ore, unit.cargo.water, unit.cargo.metal, unit.power]
            for i, a in enumerate(amounts):
                valid_actions["unit_act"]["transfer"]['resource'][i, x, y] = (a > 0)
            for direction in range(1, len(move_deltas)):
                target_pos = unit.pos + move_deltas[direction]

                # always forbid to transfer to self
                if direction == 0:
                    continue

                there_is_a_target = False
                if (target_pos >= 0).all() and (target_pos < env_cfg.map_size).all():
                    there_is_a_target = (unit_map[target_pos[0], target_pos[1]] != -1)
                if factory_under_unit(target_pos, game_state.factories[player]) is not None:
                    there_is_a_target = True

                if there_is_a_target:
                    valid_actions["unit_act"]["transfer"]["direction"][direction, x, y] = True

            # valid pickup
            valid_actions["unit_act"]["pickup"]['repeat'][0, x, y] = True
            factory = factory_under_unit(unit.pos, game_state.factories[player])
            if factory is not None:
                valid_actions["unit_act"]["act_type"][UnitActType.PICKUP, x, y] = True
                amounts = [
                    factory.cargo.ice, factory.cargo.ore, factory.cargo.water, factory.cargo.metal, factory.power
                ]
                for i, a in enumerate(amounts):
                    valid_actions["unit_act"]["pickup"]['resource'][i, x, y] = (a > 0)

            # valid dig
            if factory_under_unit(unit.pos, game_state.factories[player]) is None \
                and unit.power - action_queue_cost >= unit.unit_cfg.DIG_COST:
                if (board.lichen[x, y] > 0) or (board.rubble[x, y] > 0) \
                    or (board.ice[x, y] > 0) or (board.ore[x, y] > 0):
                    valid_actions["unit_act"]["dig"]['repeat'][:, x, y] = True

            # valid selfdestruct
            if unit.power - action_queue_cost >= unit.unit_cfg.SELF_DESTRUCT_COST:
                # self destruct can not repeat
                valid_actions["unit_act"]["self_destruct"]['repeat'][0, x, y] = True

            # valid recharge
            valid_actions["unit_act"]["recharge"]['repeat'][0, x, y] = True

        # calculate va for the flattened action space
        move_va = valid_actions["unit_act"]["move"]
        move_va = valid_actions["unit_act"]["act_type"][UnitActType.MOVE][None, None] \
                & move_va['direction'][:, None] \
                & move_va['repeat'][None, :]  # 5*2=10

        transfer_va = valid_actions["unit_act"]["transfer"]
        transfer_va = valid_actions["unit_act"]["act_type"][UnitActType.TRANSFER][None, None, None] \
                & transfer_va['direction'][:, None, None] \
                & transfer_va['resource'][None, :, None] \
                & transfer_va['repeat'][None, None, :] # 5*5*2=50

        pickup_va = valid_actions["unit_act"]["pickup"]
        pickup_va = valid_actions["unit_act"]["act_type"][UnitActType.PICKUP][None, None] \
                & pickup_va['resource'][:, None] \
                & pickup_va['repeat'][None, :] # 5*2=10

        dig_va = valid_actions["unit_act"]["act_type"][UnitActType.DIG][None] \
            & valid_actions["unit_act"]["dig"]['repeat']  # 2

        self_destruct_va = valid_actions["unit_act"]["act_type"][UnitActType.SELF_DESTRUCT][None] \
            & valid_actions["unit_act"]["self_destruct"]['repeat']  # 2

        recharge_va = valid_actions["unit_act"]["act_type"][UnitActType.RECHARGE][None] \
            & valid_actions["unit_act"]["recharge"]['repeat']  # 2

        do_nothing_va = valid_actions["unit_act"]["act_type"][UnitActType.DO_NOTHING]  # 1

        valid_actions = {}
        if not EnvParam.rule_based_early_step:
            if game_state.env_steps == 0:
                bid_va = np.ones(ActDims.bid, dtype=np.bool8)
            else:
                bid_va = np.zeros(ActDims.bid, dtype=np.bool8)

            if game_state.env_steps != 0 \
                and game_state.real_env_steps < 0 \
                and my_turn_to_place_factory(game_state.teams[player].place_first, game_state.env_steps):
                factory_spawn = board.valid_spawns_mask
            else:
                factory_spawn = np.zeros_like(board.valid_spawns_mask, dtype=np.bool8)

            valid_actions = {
                "bid": bid_va,
                "factory_spawn": factory_spawn,
            }

        valid_actions.update({
            "factory_act": factory_va,
            "move": move_va,
            "transfer": transfer_va,
            "pickup": pickup_va,
            "dig": dig_va,
            "self_destruct": self_destruct_va,
            "recharge": recharge_va,
            "do_nothing": do_nothing_va,
        })
        return valid_actions

    def action_stats(self, player, actions, game_state: GameState):
        action_stats = {
            'bid_sum': 0,
            'spawn_water_sum': 0,
            'spawn_metal_sum': 0,
            'build_light_cnt': 0,
            'build_heavy_cnt': 0,
            'water_cnt': 0,
            'factory_do_nothing_cnt': 0,
            'move_cnt': 0,
            'transfer_cnt': 0,
            'pickup_cnt': 0,
            'dig_cnt': 0,
            'self_destruct_cnt': 0,
            'recharge_cnt': 0,
            'robot_do_nothing_cnt': 0,
            'transfer_ice': [],
            'transfer_ore': [],
            'transfer_water': [],
            'transfer_metal': [],
            'transfer_power': [],
            'pickup_ice': [],
            'pickup_ore': [],
            'pickup_water': [],
            'pickup_metal': [],
            'pickup_power': [],
        }
        factory_actions = [
            'build_light_cnt',
            'build_heavy_cnt',
            'water_cnt',
            'factory_do_nothing_cnt',
        ]
        unit_actions = [
            'move_cnt',
            'transfer_cnt',
            'pickup_cnt',
            'dig_cnt',
            'self_destruct_cnt',
            'recharge_cnt',
        ]

        action_stats['bid_sum'] = actions.get('bid', 0)
        action_stats['spawn_water_sum'] = actions.get('water', 0)
        action_stats['spawn_metal_sum'] = actions.get('metal', 0)

        for factory_id in game_state.factories[player].keys():
            if factory_id not in actions:
                action_stats['factory_do_nothing_cnt'] += 1
        for unit_id in game_state.units[player].keys():
            if unit_id not in actions:
                action_stats['robot_do_nothing_cnt'] += 1

        for id, action in actions.items():
            if id.startswith('factory_'):
                action_stats[factory_actions[action]] += 1
            elif id.startswith('unit_'):
                for act in action:
                    act_type = UnitActType(act[0])
                    action_stats[unit_actions[act_type]] += 1
                    if act_type in [UnitActType.TRANSFER, UnitActType.PICKUP]:
                        resource = ResourceType(act[2])
                        amount = act[3]
                        action_stats[f"{act_type.name.lower()}_{resource.name.lower()}"].append(amount)

        action_stats['transfer_ice'] = np.array(action_stats['transfer_ice'])
        action_stats['transfer_ore'] = np.array(action_stats['transfer_ore'])
        action_stats['transfer_water'] = np.array(action_stats['transfer_water'])
        action_stats['transfer_metal'] = np.array(action_stats['transfer_metal'])
        action_stats['transfer_power'] = np.array(action_stats['transfer_power'])
        action_stats['pickup_ice'] = np.array(action_stats['pickup_ice'])
        action_stats['pickup_ore'] = np.array(action_stats['pickup_ore'])
        action_stats['pickup_water'] = np.array(action_stats['pickup_water'])
        action_stats['pickup_metal'] = np.array(action_stats['pickup_metal'])
        action_stats['pickup_power'] = np.array(action_stats['pickup_power'])

        return action_stats
