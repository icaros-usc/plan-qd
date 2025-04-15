from src.measures.utilities import scrub_movement, get_loc, check_obj_changes, find_dist, as_list
import numpy as np
from operator import add


def calc_state_summary(json_df):
    state_summary = []
    for index in range(len(json_df)):
        temp = []
        for player in json_df[index]['players']:
            if 'held_object' in player.keys():
                if player['held_object'] is None:
                    temp += [list(player['position']) + [player['held_object']]]
                else:
                    temp += [list(player['position']) + [player['held_object']['name']]]
            else:
                temp += [list(player['position']) + [None]]

        for item in json_df[index]['objects']:
            temp_object = []
            temp_object += [item['name']]
            temp_object += [item['position']]
            if 'state' in item.keys():
                temp_object += [item['state']]
            elif '_ingredients' in item.keys():
                temp_object += [['ingredients', len(item['_ingredients']), 0]]
            else:
                temp_object += [None]
            temp += [temp_object]
        state_summary += [temp]
    return state_summary


# Calculate Updates
def calc_updates(state_summary):
    updates = []
    past = state_summary[0]
    movements = 0

    updates += [[[0, movements]] + past]
    for index in range(1, len(state_summary)):
        current = state_summary[index]
        if current != past:
            if scrub_movement(current) != scrub_movement(past):
                updates += [[[index, movements]] + current]
                movements = 0
            else:
                movements += 1
        past = current
    return updates


def in_item_list(item_list, target, stop):
    for item in item_list[0:stop]:
        if target[1:] == item[1:]:
            return True


# Calculate id's
def calc_ids(updates):
    item_list = [[]]
    obj_id = 0

    for i in range(1, len(updates)):
        for cur_loc in range(3, len(updates[i])):
            id_loc = get_loc(updates[i - 1], updates[i][cur_loc])
            if id_loc is None:
                id_loc = get_loc(updates[i - 1], updates[i][cur_loc][0:2] + [not updates[i][cur_loc][2]])
            if id_loc is not None:
                updates[i][cur_loc].append(updates[i - 1][id_loc][-1])

        lost_objs = check_obj_changes(i, i - 1, updates)
        gained_objs = check_obj_changes(i - 1, i, updates)

        # Check if players picked up or dropped an item
        for j in range(2, 0, -1):
            if updates[i - 1][j][2] != updates[i][j][2]:
                if updates[i - 1][j][2] is None:
                    gen_id = True
                    for obj in lost_objs:
                        if obj[0] == updates[i][j][2] and find_dist(updates[i][j][0:2], obj[1]) < 2:
                            index_loc = get_loc(updates[i - 1], obj)
                            updates[i][j].append(updates[i - 1][index_loc][-1])
                            item_list.append(
                                [updates[i][0], updates[i - 1][index_loc][-1], j, updates[i - 1][index_loc][0]])
                            # Preserve ID of picked up item.
                            # Remove item
                            lost_objs.remove(obj)
                            gen_id = False
                            break
                    # Generate new ID for gained item.
                    if gen_id:
                        updates[i][j].append(obj_id)
                        item_list.append([updates[i][0], obj_id, j, updates[i][j][2]])
                        obj_id += 1
                elif updates[i][j][2] is None:
                    if len(gained_objs) == 0:
                        if updates[i - 1][j][2] == 'soup':
                            item_list.append([updates[i][0], updates[i - 1][j][-1], j, 'served_soup'])
                            pass
                        elif updates[i - 1][j][2] == 'boiled_chicken':
                            item_list.append([updates[i][0], updates[i - 1][j][-1], j, 'served_boiled_chicken'])
                            pass
                        elif updates[i - 1][j][2] == 'steak':
                            item_list.append([updates[i][0], updates[i - 1][j][-1], j, 'served_steak'])
                            pass
                        elif updates[i - 1][j][2] == 'steak_onion':
                            item_list.append([updates[i][0], updates[i - 1][j][-1], j, 'served_steak_onion'])
                            pass
                        elif updates[i - 1][j][2] == 'boiled_chicken_onion':
                            item_list.append([updates[i][0], updates[i - 1][j][-1], j, 'served_boiled_chicken_onion'])
                            pass
                        else:  # Soup Logic - Check only if current object is soup, and the player placed an item.
                            for obj in updates[i][3:]:
                                if obj[0] == "soup" and find_dist(updates[i][j][0:2], obj[1]) < 2:
                                    soup_loc = get_loc(updates[i - 1], obj)
                                    if soup_loc is not None:
                                        prior_count = updates[i - 1][soup_loc][2][1]
                                        if obj[2][1] > prior_count:
                                            prior_id = obj[-1]
                                            merged_id = as_list(updates[i - 1][j][-1]) + as_list(prior_id)
                                            obj[-1] = merged_id
                                            item_list.append([updates[i][0], merged_id, j, updates[i - 1][j][2]])
                                    else:
                                        if obj[2][1] > len(as_list(obj[-1])):
                                            prior_id = obj[-1]
                                            merged_id = as_list(updates[i - 1][j][-1]) + as_list(prior_id)
                                            obj[-1] = merged_id
                                            item_list.append([updates[i][0], merged_id, j, updates[i - 1][j][2]])
                    else:
                        for obj in gained_objs:
                            if find_dist(updates[i][j][0:2], obj[1]) < 2:
                                if obj[0] == updates[i - 1][j][2]:
                                    index_loc = get_loc(updates[i], obj)
                                    updates[i][index_loc].append(updates[i - 1][j][-1])
                                    gained_objs.remove(obj)

                                elif obj[0] == "soup" and find_dist(updates[i][j][0:2], obj[1]) < 2:
                                    index_loc = get_loc(updates[i], obj)
                                    updates[i][index_loc].append(updates[i - 1][j][-1])
                                    gained_objs.remove(obj)
                                elif obj[0] == "boiled_chicken" and updates[i - 1][j][2] == 'chicken':
                                    index_loc = get_loc(updates[i], ['boiled_chicken', obj[1]])
                                    updates[i][index_loc].append(updates[i - 1][j][-1])
                                    gained_objs.remove(obj)
                                elif obj[0] == "clean_plate" and updates[i - 1][j][2] == 'dirty_plate':
                                    index_loc = get_loc(updates[i], ['clean_plate', obj[1]])
                                    updates[i][index_loc].append(updates[i - 1][j][-1])
                                    gained_objs.remove(obj)
                                elif obj[0] == "steak" and updates[i - 1][j][2] == 'meat':
                                    index_loc = get_loc(updates[i], ['steak', obj[1]])
                                    updates[i][index_loc].append(updates[i - 1][j][-1])
                                    gained_objs.remove(obj)
                                elif obj[0] == "garnish" and updates[i - 1][j][2] == 'onion':
                                    index_loc = get_loc(updates[i], ['garnish', obj[1]])
                                    updates[i][index_loc].append(updates[i - 1][j][-1])
                                    gained_objs.remove(obj)

                elif updates[i - 1][j][2] == 'dish':
                    if updates[i][j][2] != 'dish':
                        prior_id = updates[i - 1][get_loc(updates[i - 1], lost_objs[0])][-1]
                        merged_id = as_list(updates[i - 1][j][-1]) + as_list(prior_id)
                        updates[i][j].append(merged_id)
                        item_list.append([updates[i][0], merged_id, j, updates[i][j][2]])

                elif updates[i - 1][j][2] == 'clean_plate':
                    for obj in lost_objs:
                        if obj[0] == updates[i][j][2]:
                            prior_id = updates[i - 1][get_loc(updates[i - 1], obj)][-1]
                            merged_id = as_list(updates[i - 1][j][-1]) + as_list(prior_id)
                            updates[i][j].append(merged_id)
                            item_list.append([updates[i][0], merged_id, j, updates[i][j][2]])

                elif updates[i][j][2][-6:] == '_onion':
                    if updates[i][j][2][:-6] == updates[i-1][j][2]:
                        for obj in lost_objs:
                            if obj[0] == 'garnish':
                                prior_id = updates[i - 1][get_loc(updates[i - 1], obj)][-1]
                                merged_id = as_list(updates[i - 1][j][-1]) + as_list(prior_id)
                                updates[i][j].append(merged_id)
                                item_list.append([updates[i][0], merged_id, j, updates[i][j][2]])



            elif updates[i][j][2] is not None:
                updates[i][j].append(updates[i - 1][j][-1])
    return updates, item_list[1:]


def calc_statistics(item_list, time_coef=0.15):
    cur_inst = 0
    p1_actions = [0, 0, 0, 0]
    p2_actions = [0, 0, 0, 0]

    prior_action = item_list[0]
    wasted_actions_strict = []
    wasted_actions = []
    i = 0

    times = []
    movement = [0, 0]

    id_actions = {}

    for action in item_list:
        # Measures individual performance
        times += [(action[0][0] - cur_inst) * time_coef]
        cur_inst = action[0][0]
        if action[2] == 1:
            movement[0] += action[0][1]
            if action[3] in ['onion', 'meat', 'chicken']:
                p1_actions[0] += 1
            elif action[3] == ['dish', 'dirty_plate', 'clean_plate']:
                p1_actions[1] += 1
            elif action[3] == ['soup', 'steak', 'boiled_chicken', 'boiled_chicken_onion', 'steak_onion']:
                p1_actions[2] += 1
            elif action[3][:6] == 'served':
                p1_actions[3] += 1
        elif action[2] == 2:
            movement[1] += action[0][1]
            if action[3] == ['onion', 'meat', 'chicken']:
                p2_actions[0] += 1
            elif action[3] == 'dish':
                p2_actions[1] += 1
            elif action[3] == 'soup':
                p2_actions[2] += 1
            elif action[3][:6] == 'served':
                p2_actions[3] += 1

        hashed_id = tuple(as_list(action[1]))
        # Measures collab on specific items.
        if hashed_id not in id_actions.keys():
            id_actions[hashed_id] = [[0, 0], action[3]]

        if action[2] == 1:
            id_actions[hashed_id][0][0] += 1
        else:
            id_actions[hashed_id][0][1] += 1

        if action[3][:6] == 'served':
            id_actions[hashed_id][1] = 'served'

        # Measures wasted actions
        i += 1
        if action[1:] == prior_action[1:]:
            wasted_actions += [action[0][0]]
        if in_item_list(item_list, action, i):
            wasted_actions_strict += [action[0][0]]
        prior_action = action

    # Get contributions to each recipe by each player.
    recipe_contributions = {}
    for key in id_actions.keys():
        if id_actions[key][1] == 'served':
            recipe_contributions[key] = [0, 0]
            for key_alt in id_actions.keys():
                if set(key_alt).issubset(key):
                    recipe_contributions[key] = list(map(add, recipe_contributions[key], id_actions[key_alt][0]))

    teamwork_list = []
    for key in recipe_contributions:
        filtered_contributions = list(filter(lambda v: v == v, recipe_contributions[key]))
        teamwork_list += [np.min(filtered_contributions) / np.sum(filtered_contributions)]

    teamwork = np.mean(teamwork_list)
    player_actions = [p1_actions, p2_actions]
    return teamwork, times, player_actions, movement, wasted_actions, wasted_actions_strict