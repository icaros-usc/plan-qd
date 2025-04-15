"""Base LLM emitter class
"""

import logging
import json

import numpy as np
import torch
from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import EmitterBase

from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class LLMEmitterBase(EmitterBase):
    """Base LLM emitter class
    """

    def __init__(
        self,
        archive: ArchiveBase,
        result_archive: ArchiveBase,
        batch_size: int,
        solution_dim: int,
        measure_names: list,
        initial_solutions: List[str],
        prompt: Dict[str, str],
        grid_config: Dict,
        mutation_type: List[str] = ["personality"],
        static: int = 0,
        
        bounds=None,
        seed=None,
    ):
        self._batch_size = batch_size
        self._result_archive = result_archive
        self._rng = np.random.default_rng(seed)

        self._initial_solutions = list(initial_solutions)
        self._static = static

        self._prompt = prompt
        self._grid_config = grid_config

        self._special_keys = ["personality"]

        self._mutation_type = mutation_type
        for m in self._mutation_type:
            if m not in self._special_keys and m not in list(self._prompt.keys()):
                raise ValueError(f"Invalid mutation type. {m} is not supported.")

        if bounds is not None:
            raise ValueError("Bounds not supported for this emitter")

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=solution_dim,
            bounds=bounds,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # stored prompts for the current batch
        self._prompts = []

        # create general templates to use for mutators (keep things consistent)
        self._start_template = f"""An AI agent is playing Overcooked with another AI agent. The agent is provided with the following game rules:

{self._prompt["rules"]}"""

        if self._grid_config is not None:
            self._start_template += f"""
The environment is structured as follows:

{self._grid_config["description"]}"""

        self.update_measures(measure_names)

    @property
    def batch_size(self):
        return self._batch_size

    @staticmethod
    def format_prompt(
        context,
        **kwargs
    ):
        for key, value in kwargs.items():
            context = context.replace(f"{{{key}}}", value)
        return context
    
    def update_measures(
        self,
        measure_names: list
    ):
        """Update the measure names from a list of measures
        """
        self._measure_names = measure_names
        self._measure_names_as_language = [m['name'].replace("_", " ") for m in self._measure_names]

    def get_num_agents(
        self
    ):
        """Returns the number of agents present in the mutation
        """
        return self._solution_dim // len(self._mutation_type)

    def _random_sample(
        self,
        num_samples: int,
        joint: bool = False # if joint is True, the solns will be formatted for joint mutation
    ):
        """Randomly samples 'num_samples' solutions. """

        if self._archive.empty:
            logging.warning("Archive is empty. Randomly sampling solutions from initial solutions.")
            return self._sample_x0(num_samples)[:num_samples]

        if joint:
            needed = num_samples
        else:
            needed = (num_samples+1) // self.get_num_agents()

        # TODO: consider more robust elite sampling
        elites = self._archive.sample_elites(needed)

        if not joint:
            # if not joint, we need to select only the top solutions
            return elites['solution'].reshape(-1, len(self._mutation_type))[:num_samples]

        return elites['solution']
    
    def _reformat_with_static_agents(
        self,
        results: np.array,
    ):
        """Reformats the solutions with static agents
        """
        static_size = self._static
        dynamic_size = self.get_num_agents() - self._static

        if dynamic_size != 0:
            results = np.array(results, dtype=self.archive.dtypes["solution"]).reshape(-1, dynamic_size * len(self._mutation_type))

        # if static prompts are needed, then add them to the results
        # ensure that the prompts are paired with the correct solutions
        if static_size > 0:
            # for every result, we need to generate 'static' number of prompts
            # we obtain these prompts from the x0 solutions
            needed = self._batch_size * self._static
            static_results = self._sample_x0(needed)[:needed]

            if dynamic_size == 0:
                results = np.array(static_results, dtype=self.archive.dtypes["solution"]).reshape(-1, static_size * len(self._mutation_type))
            else:
                # static results are batch * static_size, and results are batch * dynamic_size, and neither are empty
                # we need to combine them in a way that the static results are interleaved with the dynamic results

                # tacky way to combine the results, but it works
                arr = []
                for res1, res2 in zip(static_results, results):
                    x = list(res1)
                    x.extend(list(res2))
                    arr.append(x)
                
                results = np.array(arr, dtype=self.archive.dtypes["solution"]).reshape(-1, self._solution_dim)

        # flip the solutions so the static solutions are at the end
        self._cur_solutions = np.flip(results, axis=1)
        return self._cur_solutions

    def _sample_x0(self, num_samples: int):
        """Sample `num_samples` initial solutions from the initial solutions list.
        If the number of samples is greater than the number of initial solutions, the initial solutions will be repeated.
        """

        all_adds = []

        if num_samples < len(self._initial_solutions):
            init_sols = self._initial_solutions[:num_samples]
        else:
            init_sols = ( self._initial_solutions * num_samples )[:num_samples]

        all_adds.append(init_sols)

        for mt in self._mutation_type:
            if mt in self._special_keys:
                continue

            if self._prompt[mt] is None:
                continue

            add = [self._prompt[mt]] * num_samples
            all_adds.append(add)

        new_array = []
        for i in range(num_samples):
            d = []
            for section in all_adds:
                d.append(section[i])

            new_array.append(d)

        return np.array(new_array, dtype=self.archive.dtypes["solution"])

    def ask(
        self,
        chat_template: bool = True # if chat template is True, the prompt will be formatted for chat
    ):
        raise NotImplementedError("Ask method not implemented for LLM emitters")


    def tell(self, solution, objective, measures, add_info, **fields):
        """tell the emitter the solution and its objectve values
        
        Args:
            solution (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            measures (numpy.ndarray): ``(n, <measure space dimension>)`` array
                with the measure space coordinates of each solution.
            add_info (dict): Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        """
        raise NotImplementedError("Tell method not implemented for LLM emitters")

    def post_process(
        self,
        results: np.array,
        **kwargs
    ):
        raise NotImplementedError("Post process method not implemented for LLM emitters")