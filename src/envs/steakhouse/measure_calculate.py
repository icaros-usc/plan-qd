from typing import List
import numpy as np

# =======================================
# ======= Workload-Based Measures =======
# =======================================

def _workload_diff(workloads: List[List[dict]], key: str) -> int:
    """Helper function to calculate workload difference from workloads list.

    Args:
        workloads: First axis is n_evals, second axis is 2 (num_agents), dict at least
            contains the given key.
        key: Key to use when calculating the difference.

    Returns:
        Array of workload values.
    """

    workload_diff = []
    for eval in workloads:
        diff = eval[0][key] - eval[1][key]
        workload_diff.append(diff)

    # return round(np.median(np.array(workload_diff)))
    return np.array(workload_diff)

def diff_num_onion_chopped(metadata, agg_metadata, model):
    """Median difference in the total number of onions chopped by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_onion_chopped")

def diff_num_onion_picked(metadata, agg_metadata, model):
    """Median difference in the total number of onions picked by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_onion_picked")

def diff_num_onion_put_on_board(metadata, agg_metadata, model):
    """Median difference in the total number of onions put on board by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_onion_put_on_board")

def diff_num_meat_picked(metadata, agg_metadata, model):
    """Median difference in the total number of meats picked by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_meat_picked")


def diff_num_meat_put_on_grill(metadata, agg_metadata, model):
    """Median difference in the total number of meats picked by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_meat_put_on_grill")

def diff_num_dirty_dish_picked(metadata, agg_metadata, model):
    """Median difference in the total number of dirty dishes picked by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_dirty_dish_picked")


def diff_num_clean_dish_picked(metadata, agg_metadata, model):
    """Median difference in the total number of clean dishes picked by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_clean_dish_picked")


def diff_num_dish_served(metadata, agg_metadata, model):
    """Median difference in the total number of dishes served by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_dish_served")


def diff_num_dish_put_in_sink(metadata, agg_metadata, model):
    """Median difference in the total number of dishes put in sink by two agents during an
    episode."""
    return _workload_diff(metadata.player_workload_list, "num_dish_put_in_sink")