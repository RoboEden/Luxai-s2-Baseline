import gzip
from zipfile import ZipFile
import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time
import shutil
import polars as pl
from argparse import ArgumentParser

EPISODE_FILE_NAME = "Episodes.csv"
EPISODE_AGENTS_FILE_NAME = "EpisodeAgents.csv"
INFO_FILE = "info.json.gz"
REPLAY_POOL = "replay_pool"
TOP_REPLAY = "top_replay"
NUM_API_CALLS_TODAY = 0
BUFFER = 1
COMPETITIONS = {
    'lux-ai-2022': 45040,
    'kore-2022': 34419,
    'lux-ai-2021': 30067,
    'hungry-geese': 25401,
    'rock-paper-scissors': 22838,
    'santa-2020': 24539,
    'halite': 18011,
    'google-football': 21723,
}


def read_episodes(args):

    def open_zip_file(zip_file, **kwargs):
        entire_path = f"{args.workspace}/{zip_file}"
        if os.path.exists(entire_path):
            df = pl.read_csv(entire_path, **kwargs)
        elif os.path.exists(f"{entire_path}.zip"):
            os.system("unzip " + f"{entire_path}.zip")
            df = pl.read_csv(entire_path, **kwargs)
        else:
            raise FileExistsError
        return df
    if not os.path.exists(EPISODE_FILE_NAME):
        print('Lack of files. Please download csv files in https://www.kaggle.com/datasets/kaggle/meta-kaggle?resource=download!')
        raise FileNotFoundError
    episodes_df = open_zip_file(EPISODE_FILE_NAME)
    episodes_df = episodes_df.filter(pl.col('CompetitionId') == COMPETITIONS[args.competition])
    episodes_df = episodes_df.to_pandas()
    print(f'Episodes.csv: {len(episodes_df)} rows after filtering for {args.competition}.')

    epagents_df = open_zip_file(EPISODE_AGENTS_FILE_NAME, dtypes={'Reward': pl.Float32})
    epagents_df = epagents_df.filter(pl.col("EpisodeId").is_in(episodes_df['Id'].to_list()))
    epagents_df = epagents_df.to_pandas()
    epagents_df['InitialConfidence'] = epagents_df['InitialConfidence'].replace("", np.nan).astype(float)
    epagents_df['InitialScore'] = epagents_df['InitialScore'].replace("", np.nan).astype(float)
    print(f'EpisodeAgents.csv: {len(epagents_df)} rows after filtering for {args.competition}.')

    # Prepare dataframes
    episodes_df = episodes_df.set_index(['Id'])
    episodes_df['CreateTime'] = pd.to_datetime(episodes_df['CreateTime'])
    episodes_df['EndTime'] = pd.to_datetime(episodes_df['EndTime'])

    epagents_df.fillna(0, inplace=True)
    epagents_df = epagents_df.sort_values(by=['Id'], ascending=False)
    return episodes_df, epagents_df


def get_top_scoring(args, episodes_df: pd.DataFrame, epagents_df: pd.DataFrame):
    # Get top scoring submissions# Get top scoring submissions
    epagents_df.UpdatedScore = epagents_df.UpdatedScore.apply(lambda x: float(x) if x != '' else 0)
    epagents_df = epagents_df.drop_duplicates()
    if args.submission_max == -1:
        # epi2score = epagents_df.sort_values(
        #     by=['UpdatedScore'],
        #     ascending=False,
        # )
        mean_score = epagents_df.groupby(by='EpisodeId')['UpdatedScore'].mean().to_frame().rename(
            columns={
                "UpdatedScore": "MeanScore"
            }).reset_index()
        epagents_df = epagents_df.merge(mean_score, left_on='EpisodeId', right_on='EpisodeId', how="left")
        # .sort_values(ascending=False, )
        epagents_df = epagents_df.drop_duplicates().sort_values(ascending=False, by="MeanScore").reset_index(drop=True)
        epi2score = epagents_df.iloc[:int(args.num_top_replay) * 2]  # TOP REPLAY
        epi2score = epi2score[["EpisodeId", "UpdatedScore", "SubmissionId"]]

    else:
        # epagents_df
        epi2score = epagents_df.sort_values(
            by=['UpdatedScore'],
            ascending=False,
        ).groupby('SubmissionId').head(args.submission_max)
        epi2score = epi2score.drop_duplicates().reset_index(drop=True)
        epi2score = epi2score.iloc[:int(args.num_top_replay)]  # TOP REPLAY
    epi2score = pd.merge(left=episodes_df, right=epi2score, left_on='Id', right_on='EpisodeId')
    # SubmissionId : UpdatedScore
    sub2score_top = pd.Series(
        epi2score.UpdatedScore.values,
        index=epi2score.SubmissionId,
    ).sort_values(ascending=False).to_dict()
    sub2episodes = epi2score.groupby(by="SubmissionId")["EpisodeId"].apply(list).to_dict()
    return sub2score_top, sub2episodes, epi2score


def view_saved_replay(args, episodes_df, epagents_df):
    sub2score_top, sub2episodes, epi2score = get_top_scoring(args, episodes_df, epagents_df)
    print(f'{len(sub2score_top)} submissions with score top {args.num_top_replay}')
    candidates = len(set([item for sublist in sub2episodes.values() for item in sublist]))
    print(f'{candidates} episodes for these {len(sub2score_top)} submissions')

    if os.path.exists(f"{args.replay_dir}/{REPLAY_POOL}/{INFO_FILE}"):
        with gzip.open(f"{args.replay_dir}/{REPLAY_POOL}/{INFO_FILE}", "rb") as f:
            replay_info = json.load(f)
        seen_episodes = list(os.walk(f"{args.replay_dir}/{REPLAY_POOL}"))[0][-1]
        seen_episodes = list(
            map(lambda file_name: int(file_name.split(".")[0])
                if file_name.split(".")[0].isdigit() else None, seen_episodes))
        seen_episodes = list(filter(lambda file_id: file_id is not None, seen_episodes))
        # seen_episodes = list(replay_info.keys())
        seen_episodes = [int(submission_id) for submission_id in seen_episodes]
    else:
        replay_info = {}
        seen_episodes = []

    
    epi_remaining = np.setdiff1d(
        [item for sublist in sub2episodes.values() for item in sublist],
        seen_episodes,
    )
    print(f'{len(epi_remaining)} of these {candidates} episodes not yet saved')
    print('Total of {} games in existing library'.format(len(seen_episodes)))
    return epi_remaining, seen_episodes, epi2score


def create_info_json(epid, episodes_df, epagents_df):
    create_seconds = int((episodes_df[episodes_df.index == epid]['CreateTime'].values[0]).item() / 1e9)
    end_seconds = int((episodes_df[episodes_df.index == epid]['CreateTime'].values[0]).item() / 1e9)

    agents = [None, None]
    epagent_df = epagents_df[epagents_df['EpisodeId'] == epid].sort_values(by=['Index'])
    for _, row in epagent_df.iterrows():
        agent = {
            "id": int(row["Id"]),
            "state": int(row["State"]),
            "submissionId": int(row['SubmissionId']),
            "reward": float(row['Reward']),
            "index": int(row['Index']),
            "initialScore": float(row['InitialScore']),
            "initialConfidence": float(row['InitialConfidence']),
            "updatedScore": float(row['UpdatedScore']),
            "updatedConfidence": float(row['UpdatedConfidence']),
            "teamId": None,
        }
        agents[agent["index"]] = agent

    info = {
        "id": int(epid),
        "competitionId": int(COMPETITIONS[args.competition]),
        "createTime": {
            "seconds": int(create_seconds)
        },
        "endTime": {
            "seconds": int(end_seconds)
        },
        "agents": agents
    }

    return info


def saveEpisode(args, seen_episodes, epid_update, epi2score, episodes_df, epagents_df):
    # read match info
    if os.path.exists(f"{args.replay_dir}/{REPLAY_POOL}/{INFO_FILE}"):
        with gzip.open(f"{args.replay_dir}/{REPLAY_POOL}/{INFO_FILE}", "rb") as finfo:
            replay_info = json.load(finfo)
    else:
        replay_info = {}
    epid_not_info = np.setdiff1d(seen_episodes, list(map(lambda id: int(id), replay_info.keys()))).tolist()

    # save replay to replay_pool
    num_saved = 0
    for epid in epid_update + epid_not_info:
        epid = int(epid)
        info = create_info_json(epid, episodes_df, epagents_df)

        start_time = datetime.datetime.now()
        epid_file = f'{args.replay_dir}/{REPLAY_POOL}/{epid}.json.gz'
        if epid in epid_update:
            try:
                re = requests.post(args.url, json={"episodeId": int(epid)})
                replay = re.json()
            except ConnectionError as e:
                print(f"ConnectionError for {epid}, {e}")
                continue
            if BUFFER > (datetime.datetime.now() - start_time).seconds:
                time.sleep(BUFFER - (datetime.datetime.now() - start_time).seconds)
            if "ERROR" not in replay['statuses'] and replay:
                os.makedirs(os.path.dirname(epid_file), exist_ok=True)
                with gzip.open(epid_file, 'wb') as freplay:
                    freplay.write(bytes(json.dumps(replay), encoding='ascii'))
                    num_saved += 1
                    print(f"{num_saved} replay {epid} saved")

        if os.path.exists(epid_file):
            with gzip.open(epid_file, "rb") as gfile:
                try:
                    replay = json.load(gfile)
                    print(f"Read {epid_file} info")
                    for player in info["agents"]:
                        idx = player['index']
                        player['teamId'] = replay["info"]["TeamNames"][idx]
                    if "ERROR" not in replay['statuses']:
                        replay_info[f"{epid}"] = info
                except Exception as e:
                    print(f"{epid_file} has errors", e)
                    os.remove(epid_file)

    # save match info to replay_pool
    with gzip.open(f"{args.replay_dir}/{REPLAY_POOL}/{INFO_FILE}", 'wb') as finfo:
        finfo.write(bytes(json.dumps(replay_info), encoding='ascii'))

    # copy replay and match to top_replay
    os.makedirs(f"{args.replay_dir}/{TOP_REPLAY}/", exist_ok=True)
    shutil.rmtree(f"{args.replay_dir}/{TOP_REPLAY}/")
    top_info = {}
    for epid in epi2score["EpisodeId"].to_list():
        if os.path.exists(f'{args.replay_dir}/{REPLAY_POOL}/{epid}.json.gz'):
            os.makedirs(f'{args.replay_dir}/{TOP_REPLAY}/{epid}.json.gz', exist_ok=True)
            top_info[epid] = replay_info[f"{epid}"]
            shutil.copy2(
                f'{args.replay_dir}/{REPLAY_POOL}/{epid}.json.gz',
                f'{args.replay_dir}/{TOP_REPLAY}/{epid}.json.gz',
            )
            print(f"copy {epid}.json.gz to top_replay")
    with gzip.open(f"{args.replay_dir}/{TOP_REPLAY}/{INFO_FILE}", 'wb') as finfo:
        finfo.write(bytes(json.dumps(top_info), encoding='ascii'))


def remove_error_files(args):
    top_replay_folder = f"{args.replay_dir}/{TOP_REPLAY}"
    replay_pool_folder = f"{args.replay_dir}/{REPLAY_POOL}"
    for folder in [top_replay_folder, replay_pool_folder]:
        with gzip.open(f"{folder}/{INFO_FILE}", 'rb') as finfo:
            replay_info = json.load(finfo)
        for root, dir, files in os.walk(folder):
            for file in files:
                if os.stat(f"{root}/{file}").st_size < 100:
                    os.remove(f"{root}/{file}")
                    print(f"{file} has been remove")
                    id = int(file.split(".")[0])
                    replay_info.pop(f"{id}")
        with gzip.open(f"{folder}/{INFO_FILE}", 'wb') as finfo:
            finfo.write(bytes(json.dumps(replay_info), encoding='ascii'))


def main(args):
    episodes_df, epagents_df = read_episodes(args)
    epi_remaining, seen_episodes, epi2score = view_saved_replay(args, episodes_df, epagents_df)

    # sorted by submission score
    num_actual_update = min(args.max_calls_per_day - args.num_api_calls_today, len(epi_remaining))
    epi_remaining = epi_remaining[:num_actual_update].tolist()
    epi_available = epi_remaining + seen_episodes
    epi2score = epi2score.loc[epi2score["EpisodeId"].isin(epi_available)]
    episodes_df = episodes_df.loc[epi_available]
    epagents_df = epagents_df.loc[epagents_df["EpisodeId"].isin(epi_available)]

    saveEpisode(args, seen_episodes, epi_remaining, epi2score, episodes_df, epagents_df)
    print(f'Episodes saved: {len(epi_available)}')  # min(num_top_replay, max_calls_per_day - num_api_calls_today)

    remove_error_files(args)


if __name__ == "__main__":
    '''
    COPY FROM 
    https://www.kaggle.com/code/kuto0633/luxai2-episode-scraper-match-downloader
    '''
    parser = ArgumentParser(description="LuxAI2 Episode Scraper Match Downloader")
    parser.add_argument("--url", default="https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay")
    parser.add_argument("--num_top_replay", default=100, help="Top 100 game")
    parser.add_argument("--num_api_calls_today", default=0)
    parser.add_argument("--max_calls_per_day",
                        default=3600,
                        type=int,
                        help="Kaggle says don't do more than 3600 per day and 1 per second")
    parser.add_argument("--replay_dir", default="./replays/")
    parser.add_argument("--workspace", default="./")
    parser.add_argument("--submission_max", default=-1, help="The maximum replay amount for each submission")
    # parser.add_argument("--lowest_score_threshold", default=1200, type=int, help="Kaggle score")
    parser.add_argument("--competition", default="lux-ai-2022", type=str)

    args = parser.parse_args()
    main(args)
