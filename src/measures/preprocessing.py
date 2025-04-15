import os
import pandas as pd
from src.measures.utilities import evaluate, get_slice
from src.measures.get_ids import calc_ids, calc_state_summary, calc_updates, calc_statistics


def load_human_data(path):
    df = pd.read_pickle(path)
    return df.map(evaluate)


def calc_human_data(layouts_list, layout_key, df):
    human_data = [{}, {}, {}, {}, {}]
    for i in range(1, 100):
        game_slice = get_slice(df, i)
        if len(game_slice) > 0:
            layout = layout_key[tuple(game_slice.iloc[0, 3][::len(game_slice.iloc[0, 3]) - 1])]
            state_summary = calc_state_summary(list(game_slice['state']))
            updates = calc_updates(state_summary)
            updates_with_id, item_list = calc_ids(updates)
            total_score = game_slice.iloc[0, 16]
            human_data[layouts_list.index(layout)][i] = tuple(list(calc_statistics(item_list)) + [total_score])
        else:
            break
    return human_data


def calc_agent_data(layouts_list):
    cur_path = os.getcwd()
    agent_data = [{}, {}, {}, {}, {}]
    for target_layout in layouts_list:
        i = 0
        for file in os.listdir(cur_path + '\\data\\AI_games\\' + target_layout):
            game_slice = pd.read_json("data/AI_games/" + target_layout + '/' + file)
            state_summary = calc_state_summary(game_slice['ep_observations'][0])
            agent_updates = calc_updates(state_summary)
            updates_with_id, agent_item_list = calc_ids(agent_updates)
            total_score = sum(game_slice['ep_rewards'][0])
            agent_data[layouts_list.index(target_layout)][i] = tuple(list(calc_statistics(agent_item_list)) + [total_score])
            i += 1
    return agent_data

