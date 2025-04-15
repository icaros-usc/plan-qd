import numpy as np
import copy
from ast import literal_eval


def evaluate(val):
    if isinstance(val, str):
        if val[0] == '[' or val[0] == '{':
            return literal_eval(val.replace('null', 'None').replace('false', 'False').replace('true', 'True'))
    else:
        return val


def find_dist(loc1, loc2):
    x_dist = np.abs(loc1[0] - loc2[0])
    y_dist = np.abs(loc1[1] - loc2[1])
    return x_dist + y_dist


def as_list(x):
    if type(x) is list:
        return x
    else:
        return [x]


def scrub_movement(state):
    state_copy = copy.deepcopy(state)
    state_copy[0][0] = 0
    state_copy[0][1] = 0
    state_copy[1][0] = 0
    state_copy[1][1] = 0
    state_copy[-1][0] = 0
    state_copy[-1][1] = 0
    return state_copy


def get_change(index, df):
    players = df.iloc[index]
    interactions = [0, 0]
    for i in range(2):
        if len(players.iloc[13]['players'][i]) < len(players.iloc[8]['players'][i]):
            interactions[i] = 0
        elif len(players.iloc[13]['players'][i]) > len(players.iloc[8]['players'][i]):
            interactions[i] = 1
    return interactions


def get_loc(arr, target):
    for i in range(len(arr)):
        if isinstance(arr[i], list):
            if target[0:2] == arr[i][0:2]:
                return i
    return None


def item_in_list(arr, target, index):
    for i in range(len(arr)):
        if target == arr[i][index]:
            return arr[i]
    return None


def get_slice(df, target):
    sliced = []
    iteration = 0
    for i in range(len(df)):
        if iteration == target:
            sliced += [i]
        if df.iloc[i, 9] == -1.0:
            iteration += 1
        if iteration > target:
            break
    return df.iloc[sliced]


def check_obj_changes(index1, index2, updates):
    difference = []
    if len(updates[index2]) > 3 and len(updates[index1]) > 3:
        for val2 in range(3, len(updates[index2])):
            found = False
            for val1 in range(3, len(updates[index1])):
                if updates[index1][val1][0:2] == updates[index2][val2][0:2]:
                    found = True
            if not found:
                difference += [updates[index2][val2][0:2]]
    elif len(updates[index2]) > 3 >= len(updates[index1]):
        for val1 in range(3, len(updates[index2])):
            difference += [updates[index2][val1][0:2]]

    return difference