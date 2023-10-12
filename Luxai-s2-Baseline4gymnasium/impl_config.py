from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Dict
from typing_extensions import Literal

import tree


@dataclass
class ModelParam:
    action_emb_dim: int = 6
    action_queue_size: int = 20
    global_emb_dim: int = 10
    global_feature_dims: int = 32
    n_res_blocks: int = 2
    all_channel: int = 64

    map_channel = 30
    amount_distribution = "categorical"
    spawn_distribution = "beta"


@dataclass
class ActDims:
    factory_act: int = 4
    robot_act: int = 7
    direction: int = 5
    resource: int = 5
    amount: int = 10
    repeat: int = 2

    bid: int = 11


class UnitActChannel(IntEnum):
    TYPE = 0
    DIRECTION = 1
    RESOURCE = 2
    AMOUNT = 3
    REPEAT = 4
    N = 5


class UnitActType(IntEnum):
    MOVE = 0
    TRANSFER = 1
    PICKUP = 2
    DIG = 3
    SELF_DESTRUCT = 4
    RECHARGE = 5
    DO_NOTHING = 6

    @classmethod
    def get_value(cls, s: str):
        return cls.__members__[s.upper()]


class FactoryActType(IntEnum):
    BUILD_LIGHT = 0
    BUILD_HEAVY = 1
    WATER = 2
    DO_NOTHING = 3


class ResourceType(IntEnum):
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4


@dataclass
class UnitAct:
    act_type: int = ActDims.robot_act
    # action type 0
    move: Dict[str, int] = field(default_factory=lambda: {
        "direction": ActDims.direction,
        "repeat": ActDims.repeat,
    })

    # action type 1
    transfer: Dict[str, int] = field(default_factory=lambda: {
        "direction": ActDims.direction,
        "resource": ActDims.resource,
        "amount": ActDims.amount,
        "repeat": ActDims.repeat,
    })

    # action type 2
    pickup: Dict[str, int] = field(default_factory=lambda: {
        "resource": ActDims.resource,
        "amount": ActDims.amount,
        "repeat": ActDims.repeat,
    })

    # action type 3
    dig: Dict[str, int] = field(default_factory=lambda: {
        "repeat": ActDims.repeat,
    })

    # action type 4
    self_destruct: Dict[str, int] = field(default_factory=lambda: {
        "repeat": ActDims.repeat,
    })

    # action type 5
    recharge: Dict[str, int] = field(default_factory=lambda: {
        "amount": ActDims.amount,
        "repeat": ActDims.repeat,
    })

    # action type 6
    do_nothing: Dict = field(default_factory=lambda: {})


@dataclass
class FullAct:
    factory_act: Dict[str, int] = ActDims.factory_act
    unit_act: UnitAct = UnitAct()
    bid: int = ActDims.bid
    factory_spawn: int = 1


action_structure = tree.map_structure(lambda x: None, asdict(FullAct()))


@dataclass
class EnvParam:
    parser: Literal['sparse', 'dense', 'dense2'] = "dense"
    rule_based_early_step: bool = True

    map_size: int = 64
    MAX_FACTORIES: int = 5
    num_turn_per_cycle: int = 50
    init_from_replay_ratio: float = 1

    act_dims: ActDims = ActDims()
    act_dims_mapping: FullAct = FullAct()


@dataclass
class RewardParam:
    use_gamma_coe: bool = False
    zero_sum: bool = False
    global_reward_weight = 0.1
    win_reward_weight: float = 0. * global_reward_weight
    light_reward_weight: float = 0.1 * global_reward_weight
    heavy_reward_weight: float = 1 * global_reward_weight
    ice_reward_weight: float = 0.1 * global_reward_weight
    ore_reward_weight: float = 0 * global_reward_weight
    water_reward_weight: float = 0.1 * global_reward_weight
    metal_reward_weight: float = 0 * global_reward_weight
    power_reward_weight: float = 0.0001 * global_reward_weight
    lichen_reward_weight: float = 0.001 * global_reward_weight
    factory_penalty_weight: float = 1 * global_reward_weight
    lose_penalty_coe: float = 0.
    survive_reward_weight: float = 0.5 * global_reward_weight

# @dataclass
# class StatsRewardParam:
stats_reward_params = dict(
    action_queue_updates_total=0,
    action_queue_updates_success=0,
    consumption={
        "power": {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
        "water": 0,
        "metal": 0,
        "ore": {
            "LIGHT": 0,
            "HEAVY": 0,
        },
        "ice": {
            "LIGHT": 0,
            "HEAVY": 0,
        },
    },
    destroyed={
        'FACTORY': 0,
        'LIGHT': 0,
        'HEAVY': 0,
        'rubble': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
        'lichen': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
    },
    generation={
        'power': {
            'LIGHT': 0,
            'HEAVY': 0,
            'FACTORY': 0,
        },
        'water': 0.5,
        'metal': 0,
        'ore': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
        'ice': {
            'LIGHT': 0.1,
            'HEAVY': 0.1,
        },
        'lichen': 0,
        'built': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
    },
    pickup={
        'power': 0,
        'water': 0,
        'metal': 0,
        'ice': 0,
        'ore': 0,
    },
    transfer={
        'power': 0,
        'water': 0,
        'metal': 0,
        'ice': 0,
        'ore': 0,
    },
)


@dataclass
class RewardParam2:
    is_sparse: bool = False
    zero_sum: bool = False
    use_gamma_coe: bool = True
    global_reward_weight = 1

    win_reward_weight: float = 0 * global_reward_weight
    survive_reward_weight: float = 0 * global_reward_weight

    balance_reward_punishment = False  # robot build and destruction
    light_min = 10
    heavy_min = 3

    beat_opponent = False
    factories_opp_destroyed: float = 0
    unit_light_opp_destroyed: float = 0.01
    unit_heavy_opp_destroyed: float = 0.15
    lichen_opp_destroyed: float = 0.004
