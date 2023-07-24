import os
from pathlib import Path
import gzip
import json
from luxai_s2 import LuxAI_S2
import random
import traceback


def get_actions_from_replay(replay, replay_version: str):
    for step in replay['steps'][1:]:
        player_0, player_1 = step

        yield {'player_0': player_0['action'], 'player_1': player_1['action']}


def load_replay(env: LuxAI_S2, replay: str):
    # load json
    print(f"load replay from {replay}")
    with gzip.open(replay) as f:
        replay = json.load(f)

    # parse replay
    if 'configuration' in replay:
        # kaggle replay
        seed = replay['configuration']['seed']
        env.reset(seed=seed)
        actions = list(get_actions_from_replay(replay, replay['version']))

    return env, actions


def get_replay_list(replay_dir):
    if replay_dir is None:
        return []
    replay_dir = Path(replay_dir)
    with gzip.open(replay_dir / 'info.json.gz') as f:
        info = json.load(f)
    replays = [replay_dir / f'{ep}.json.gz' for ep in info.keys()]
    return replays


def random_init(env: LuxAI_S2, replay_dir: str):
    while True:
        try:
            replay_list = get_replay_list(replay_dir)
            replay = random.choice(replay_list)
            env, actions = load_replay(env, replay)
            n = random.randrange(0, len(actions))
            for i in range(n):
                done = env.step(actions[i])[2]
                if done['player_0']:
                    break
            if done['player_0']:
                continue
            break
        except:
            traceback.print_exc()

    return env
