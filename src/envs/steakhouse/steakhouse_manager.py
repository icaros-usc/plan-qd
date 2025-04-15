"""Provides SteakhouseManager."""

import copy
import logging
import time
import numpy as np
import warnings
import subprocess as sp
from typing import Collection, List, Dict, Optional


from hydra.utils import instantiate
from omegaconf import DictConfig

from .mdp.steakhouse_env import SteakhouseEnv
from .mdp.steakhouse_mdp import SteakhouseGridworld, SteakhouseState
from .steakhouse_module import SteakhouseModule
from .steakhouse_result import SteakhouseResult

from src.model.huggingface import BaseLLM

from .steakhouse_result import SteakhouseMetadata, SteakhouseResult, transform_list_of_dicts

logger = logging.getLogger(__name__)

# from src.qdmanager import get_gpu_memory
def get_gpu_memory():
    # credits to https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

class SteakhouseManager:
    """Manager for the steakhouse environments.

    Args:
        n_evals: Number of times to evaluate each solution during real evaluation.
            Evals are rolled out in the same batch and aggregated by the manager.
        module_cfg: Config passed to the SteakhouseModule. Can't use hydra's instantiate
            since we want to instantiate a new module for each worker.
        objective: Name of the objective function (it should be defined in
            objective_calculate.py).
        aggregation_type: Method of aggregating results. Currently ignored since QD +
            Steakhouse used a mix of aggregation for different metrics.
        measure_names: Names of the measure functions (it should be defined in
            measure_calculate.py).
        rng: Random generator. Can be set later. Uses `np.random.default_rng()` by
            default.
    """

    MIN_SCORE = 0

    def __init__(
        self,
        n_evals: int,
        module_cfg: dict,
        objective: str,
        aggregation_type: str = "mean",
        mutation_type: List[str] = ["personality"],
        inference_temperature: float = 1.1,
        measure_names: Collection[str] = [],
        rng: np.random.Generator = None,
        communication: Dict[str, str] = {},
    ):
        self.rng = rng or np.random.default_rng()
        self.n_evals = n_evals

        # set up modules folder that holds all local modules
        self.module_cfg = module_cfg
        self.modules = []

        self.aggregation_type = aggregation_type
        self.measure_names = measure_names
        self.objective = objective

        self.mutation_type = mutation_type

        self.inference_temperature = inference_temperature

        # communication module configuration
        self.communication = communication
    
    def update_measure_names(self, measure_names):
        self.measure_names = measure_names

    def _reset_all_modules(self, hard=False):
        for module in self.modules:
            module._reset_env(hard=hard)

    def get_obs_shape(self, use_simple_features: bool = False):
        mdp = SteakhouseGridworld.from_grid(
            self.module.grid,
            self.module.grid_world_config,
        )
        env = SteakhouseEnv.from_mdp(mdp, info_level=0, horizon=100)

        if use_simple_features:
            single_agent_state = mdp.simple_featurize(env.state)[0]
            return np.asarray(single_agent_state).reshape(-1).shape
        else:
            single_agent_state = mdp.lossless_state_encoding(env.state)[0]
            return np.transpose(single_agent_state, (2, 0, 1)).shape

    def evaluate(
        self,
        sols: np.ndarray,
        agents_cfg: DictConfig,
        model: BaseLLM,
        debug: bool = False,
        render: bool = False,
    ) -> List[SteakhouseResult]:
        """Pipeline that takes solutions and evaluates it. Pipeline will
        evaluate all solutions continuously in batch forward pass.

        Args:
            sols: Emitted solution.
            agents_cfg: Hydra config of the agents.
            model: Model to query for prompts.
            debug: Whether to print debug logs.

        This function will complete the setup for the environment, which is then
        passed into _run_evaluation_batch to run the evaluation.

        Returns:
            Results of the evaluation.
        """
        self.debug = debug

        # expand the solutions based on n_evals
        sols = np.repeat(sols, self.n_evals, axis=0)

        # Create agents from solutions to pass them for evaluation
        agents_list = []
        for sol in sols:
            agents = []

            indx = 0
            for a_cfg in agents_cfg:
                # construct prompt overrides
                prompt_overrides = {}
                for mt, val in zip(self.mutation_type, sol[(indx * len(self.mutation_type)) : (indx + 1) * len(self.mutation_type)]):
                    prompt_overrides[mt] = val

                agent = instantiate(
                    a_cfg["type"],
                    prompt_overrides=prompt_overrides,
                )
                agents.extend([copy.deepcopy(agent) for _ in range(a_cfg["num"])])

                indx += 1
            agents_list.append(agents)

        mut_type_total = len(self.mutation_type)

        if self.debug:
            logger.info("-"*20)
            # set the context for each agent
            for i in range(len(agents_list)):
                count = 0
                for j in range(len(agents_list[i])):
                    if type(agents_list[i][j]).__name__ == "LLMAgent":
                        context = sols[i][count*mut_type_total]
                        logger.info("Personality for module %d, agent %d:", i, j)
                        logger.info(context)

                        count += 1
            logger.info("-"*20)


        # Make each solution evaluation have a different seed.
        evaluation_seeds = self.rng.integers(
            np.iinfo(np.int32).max / 2, size=len(sols), endpoint=True
        )

        # create modules if they don't exist yet
        if len(self.modules) != len(agents_list):            
            start_time = time.time()
            self.modules = [
                SteakhouseModule(
                    **self.module_cfg,
                    communication = self.communication
                ) for _ in range(len(agents_list))
            ]

            if self.debug:
                logger.info("Preprocess took %f seconds", time.time() - start_time)

        self._run_evaluation_batch(agents_list, evaluation_seeds, model, render=render)

        # obtain all steakhouse metadata
        metadata = []
        results = []

        for i, module in enumerate(self.modules):
            agents = agents_list[i]

            raw_data = module.get_raw_data(agents)
            metadata.append(raw_data)

            if len(metadata) == self.n_evals:
                # compile the `n_evals` metadata into one metadata object
                transformed = transform_list_of_dicts(metadata)

                # reformat the agent data to be List[List[Dict[List]]]
                modified_agent_data = []
                for _ in range(len(agents_cfg)):
                    modified_agent_data.append([])

                for run in transformed["agent_data"]:
                    for agent_data in run:
                        indx = agent_data["agent_index"]
                        modified_agent_data[indx].extend([agent_data])

                transformed["agent_data"] = modified_agent_data

                amalgamated_metadata = SteakhouseMetadata(
                    **transformed
                )
                
                results.append(
                    SteakhouseResult.from_raw(
                        metadata=amalgamated_metadata,
                        opts={
                            "aggregation": self.aggregation_type,
                            "measure_names": self.measure_names,
                            "objective": self.objective
                        },
                        model=model
                    )
                )

                # clear the metadata for the next set of evaluations
                metadata = []
        
        # reset all the modules
        self._reset_all_modules()

        return results

    def _run_evaluation_batch(
        self,
        agents_list,
        evaluation_seeds,
        model,
        render: Optional[bool] = False
    ):
        """Internal function to run evaluation batch after setup is complete
        """
        # get scenarios and agents in batch
        num_scen_in_batch = len(agents_list)
        num_agents_per_scen = len(agents_list[0])

        # set index and mdp for each agent
        for j in range(num_scen_in_batch):
            scen = agents_list[j]
            for idx, agent in enumerate(scen):

                mdp = self.modules[j].mdp
                mlam = self.modules[j].mlam
                prompt_builder = self.modules[j].prompt_builder
                agent.set_mdp_and_mlam(mdp, mlam, prompt_builder=prompt_builder)
                agent.set_agent_index(idx)

        # NOTE: n_evals is no longer used here. all evaluations are done in a single batch
        # and results are later aggregated
        done_mask = [False] * num_scen_in_batch
        ts = 0

        # while all evals are not done, continue :)
        while not np.sum(done_mask) == num_scen_in_batch:
            # only log prompts for batching if the eval is not completed
            
            prompt_batch = []
            num_prompts_per = [] # xref for prompt evals
            for eval_idx in range(num_scen_in_batch):
                if done_mask[eval_idx]:
                    continue

                # step all active environments and log metrics
                module = self.modules[eval_idx]

                messages_to_send = []

                # step and log metrics for all agents
                while not module.done and len(messages_to_send) == 0:
                    messages_to_send = module.step_and_log_metrics(agents_list[eval_idx], render=render)
                    done_mask[eval_idx] = module.done

                module.inc_queries(len(messages_to_send))

                if not module.done and len(messages_to_send) > 0:
                    prompt_batch.extend(messages_to_send)
                    num_prompts_per.append(len(messages_to_send))

            if len(prompt_batch) != 0:
                if self.debug:
                    logger.info(f"Step {ts} with {len(prompt_batch)} updates ({np.sum(done_mask)} modules done)")

                # model will resolve splitting large batches into sub-batches
                batch_response = model.batch_query_messages(prompt_batch, temp=self.inference_temperature)

                # export metrics from the batch response
                batch_results = []

                # reformat the batch responses to the correct agents and corresponding MLAM
                trav = 0
                for num_prompts in num_prompts_per:
                    batch_results.append(batch_response[trav:trav+num_prompts])
                    trav += num_prompts

                idx = 0
                for eval_idx in range(num_scen_in_batch):
                    mod = self.modules[eval_idx]
                    if done_mask[eval_idx]:
                        continue

                    # ensure the correct results are passed from the batch results
                    mod.update_agents_ml_action(batch_results[idx])
                    idx += 1

            ts += 1

    def actual_qd_score(self, objs: List):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= self.MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed levels.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution": sol,
            "log_message": result.log_message,
        }
        return failed_level_info
