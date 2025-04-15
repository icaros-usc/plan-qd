from typing import List
import numpy as np
import math


def fitness(metadata, agg_metadata, model, **kwargs):
    """Calculate the fitness metric.
    """
    return agg_metadata.fitness

def sparse_fitness(metadata, agg_metadata, model, **kwargs):
    """Calculate the sparse fitness metric.
    """
    return agg_metadata.total_sparse_reward