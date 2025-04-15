"""Main file for handling config

To run an experiment, in the main directory run:
    python -m src.qdllm

The project is using an unofficial version of pyribs. Download with the following:
    pip install git+https://github.com/icaros-usc/pyribs.git

"""

import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.qdmanager import Manager
from src.utils.hydra_utils import define_resolvers

from transformers import set_seed
from transformers import enable_full_determinism

from argparse import ArgumentParser

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Parses command line flags and sets up and runs experiment.

    Args:
        cfg: Hydra config. See config/config.yaml and the sub-directories for default
            options and docs.
    """
    # Try, except to log any tracebacks to file
    try:
        # Define OmegaConf resolvers.
        define_resolvers()

        # Set seeds.
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # can specify deterministic=True, but there are some issues with it
        set_seed(cfg["seed"])

        # Run the experiment with manager.
        manager = Manager(seed=cfg["seed"], cfg=cfg["manager"])
        manager.run_experiment()
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
