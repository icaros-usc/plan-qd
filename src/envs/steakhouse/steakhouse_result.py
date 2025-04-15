"""Class representing the results of an evaluation."""

import logging
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from . import measure_calculate
from . import objective_calculate

logger = logging.getLogger(__name__)

def transform_list_of_dicts(list_of_dicts):
    # Initialize an empty dictionary to hold the aggregated lists
    aggregated_dict = {}

    # Iterate over the list of dictionaries
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in aggregated_dict:
                aggregated_dict[key] = []
            aggregated_dict[key].append(value)
    
    return aggregated_dict

def maybe_mean(arr, indices=None):
    """Calculates mean of arr[indices] if possible.

    indices should be a list. If it is None, the mean of the whole arr is taken.
    """
    indices = slice(len(arr)) if arr is not None and indices is None else indices
    return None if arr is None else np.mean(arr[indices], axis=0)


def maybe_median(arr, indices=None):
    """Same as maybe_mean but with median."""
    indices = slice(len(arr)) if arr is not None and indices is None else indices
    return None if arr is None else np.median(arr[indices], axis=0)


def maybe_std(arr, indices=None):
    """Same as maybe_mean but with std."""
    indices = slice(len(arr)) if arr is not None and indices is None else indices
    return None if arr is None else np.std(arr[indices], axis=0)


@dataclass
class SteakhouseMetadata:
    """Metadata obtained by running Steakhouse games"""

    fitness_list: List = None
    total_sparse_reward_list: List = None
    joint_actions_list: List = None
    agent_data: List = None # agent specific internal data
    chat_history: List = None # chat history between agents
    deliver_history: List = None # delivery history of the agents
    response_history: List = None # response history of the agents
    state_history: List = None # state history of the agents
    player_workload_list: List = None # player workload of the agents
    negotiated_strategies: List = None # negotiated strategies of the agents
    queries: List = None # agent queries


@dataclass
class AggSteakhouseMetadata:
    """Aggregate metadata obtained by running Steakhouse games"""

    fitness: float = None
    total_sparse_reward: float = None


@dataclass
class SteakhouseResult:  # pylint: disable = too-many-instance-attributes
    """Represents `n` results from an objective function evaluation.

    `n` is typically the number of evals (n_evals).

    Different fields are filled based on the objective function.
    """

    # Raw data
    metadata: dict = None

    # Aggregate data
    agg_obj: float = None
    agg_measures: np.ndarray = None  # (measure_dim,) array
    agg_metadata: dict = None

    # Measures of spread
    std_obj: float = None
    std_measure: np.ndarray = None  # (measure_dim,) array

    # Other data
    failed: bool = False
    log_message: str = None

    # LLM
    model = None

    @staticmethod
    def from_raw(
        metadata: SteakhouseMetadata,
        opts: dict = None,
        model = None
    ):
        """Constructs a OvercookedResult from raw data.

        `opts` is a dict with several configuration options. Options
        in `opts` are:
            `measure_names`: Names of the measures to return
            `objective`: Name of the objective to return
            `aggregation` (default="mean"): How each piece of data should be
                aggregated into single values. Options are:
                - "mean": Take the mean, e.g. mean measure
                - "median": Take the median, e.g. median measure (element-wise)
        """
        # Handle config options.
        opts = opts or {}
        if "measure_names" not in opts:
            raise ValueError("opts should contain `measure_names`")
        if "objective" not in opts:
            raise ValueError("opts should contain `objective`")

        opts.setdefault("aggregation", "median")

        agg_metadata = SteakhouseResult._aggregate_metadata(
            metadata, opts["aggregation"]
        )

        measures = SteakhouseResult._obtain_measure_values(
            metadata, agg_metadata, opts["measure_names"], model
        )

        # aggregate the measures based on the aggregation type
        agg_measures = np.median(measures, axis=0)

        agg_obj = SteakhouseResult._obtain_objective_values(
            metadata, agg_metadata, opts["objective"], model, measures
        )

        return SteakhouseResult(
            metadata=asdict(metadata),
            agg_obj=agg_obj,
            agg_measures=agg_measures,
            agg_metadata=asdict(agg_metadata)
        )

    @staticmethod
    def _aggregate_metadata(metadata, aggregation=None):
        if aggregation is not None:
            logger.warning("Aggregation type is currently ignored for Steakhouse.")

        # we take the median fitness performance to allow for more robustness
        agg_fitness = np.median(metadata.fitness_list)
        agg_total_sparse_reward = np.median(metadata.total_sparse_reward_list)

        return AggSteakhouseMetadata(
            fitness=agg_fitness,
            total_sparse_reward=agg_total_sparse_reward,
        )

    @staticmethod
    def _obtain_measure_values(metadata, agg_metadata, measure_names, model):
        measures = []
        for measure_name in measure_names:
            if type(measure_name) == dict:
                measure_name = measure_name['name']
            
            measure_fn = getattr(measure_calculate, measure_name)
            measure_val = measure_fn(metadata, agg_metadata, model)
            measures.append(measure_val)

        return np.array(measures).T

    @staticmethod
    def _obtain_objective_values(metadata, agg_metadata, objective_name, model, measures):
        objective_fn = getattr(objective_calculate, objective_name)
        return objective_fn(metadata, agg_metadata, model, measures=measures)