"""Random emitter that generates random solutions.

Used as a baseline for comparison.
"""

import logging
import json

import numpy as np
from ribs.archives import ArchiveBase
import re
from ._llm_emitter import LLMEmitterBase

from typing import List, Dict

logger = logging.getLogger(__name__)

class RandomBatchEmitter(LLMEmitterBase):
    """Random emitter that generates random solutions.
    """

    def __init__(
        self,
        archive: ArchiveBase,
        result_archive: ArchiveBase,
        solution_dim: int,
        measure_names: list,
        initial_solutions: List[str],
        prompt: Dict[str, str],
        grid_config: Dict,
        mutation_type: List[str],
        batch_size: int,
        static: int = 0,
        bounds=None,
        seed=None,
    ):

        LLMEmitterBase.__init__(
            self,
            archive,
            result_archive=result_archive,
            batch_size=batch_size,
            solution_dim=solution_dim,
            measure_names=measure_names,
            initial_solutions=initial_solutions,
            prompt=prompt,
            grid_config=grid_config,
            mutation_type=mutation_type,
            static=static,
            bounds=bounds,
            seed=seed
        )

        # internal results for the emitter to learn the underlying distribution
        self._internal_results = []

    def ask(
        self,
        chat_template = True, # if chat template is True, the prompt will be formatted for chat
    ):
        """Random emitter simply generates random mutations for a given prompt
        """

        num_prompts_to_mutate = self._batch_size * (self._solution_dim - self._static)
        if num_prompts_to_mutate == 0:
            return np.array([], dtype=self.archive.dtypes["solution"])
        
        # override solution sampling to complete this multiple times in the case of errors
        num_prompts_to_mutate = 10 

        # add prompts to the list
        random_prompts = []
        for sol in range(num_prompts_to_mutate):
            # personality = sol[0]
            x0 = self._initial_solutions[0]

            new_random_prompt = f"""{self._start_template}

The agent currently has the following personality:

{x0}

Create {self._batch_size * 2} random personalities for the agent to play the game optimally with a random strategy. Ensure the new personality is in second person. Keep the new personalities brief and to the point. 

Provide the new personalities in JSON format as a list of strings."""

            random_prompts.append(new_random_prompt)

        # return formatted prompt
        self._prompts = np.array(random_prompts, dtype=self.archive.dtypes["solution"])
        return self._prompts


    def tell(self, solution, objective, measures, add_info, **fields):
        logger.info(f"RandomEmitter does not require a tell method.")
        pass

    def post_process(
        self,
        results: np.array,
        **kwargs
    ):
        """Since random emitter results does not require formatting, we
        simply reformat and return the results"""
        
        new_results = []
        
        # try each result
        for r in results:
            try:
                # get only what is between the brackets
                x = re.search(r'```(?:json)?\n(\[.*?\])\n```', r, re.DOTALL)

                # load the json
                jsload = json.loads(x.group(1))

                for j in jsload:
                    if type(j) == str:
                        new_results.append(j)

                if len(new_results) >= self._batch_size * 2:
                    break
            except Exception as e:
                print("Failed to parse! Please reload the experiment")

        # keep results to multiple of 2
        new_results = new_results[:(len(new_results) - (len(new_results) % 2))]

        return self._reformat_with_static_agents(new_results)