from .scheduler import Scheduler

from hydra.utils import instantiate

from typing import Dict

import numpy as np

from copy import deepcopy

class LLMScheduler(Scheduler):
    def __init__(
        self,
        model, 
        # prompt: Dict[str, str],
        **kwargs
    ):
        self._init_kwargs = kwargs
        super(LLMScheduler, self).__init__(**kwargs)

        # set the model here (created by hydra instantiate)
        self._cur_prompts = []
        self._model = model

    def set_model(self, model):
        self._model = model

    def get_model(self):
        return self._model
    
    def serializable_copy(self):
        """Return a deep copy of the scheduler, without the LLM model"""
        sched = LLMScheduler(
            None,
            # self._prompt,
            **self._init_kwargs
        )

        sched._cur_solutions = self._cur_solutions
        sched._cur_prompts = self._cur_prompts

        return sched

    def remove_unserializable(self):
        """Removes unserializable content for the scheduler
        so that it can be saved for reloading
        """
        self._model = None
    
    def ask(self):
        """Implementation of LLM logic ontop of default pyribs scheduler.

        The scheduler first calls the parent model to generate a set of
        solutions. It then takes these solutions and runs batch inference
        on them given the model specified in the scheduler model_cfg.

        If joint mutation is specified, the scheduler will try to mutate
        the two solutions jointly, as opposed to independently.
        """

        if self._last_called in ["ask", "ask_dqd"]:
            raise RuntimeError(
                "ask cannot be called immediately after " + self._last_called
            )
        self._last_called = "ask"

        self._cur_prompts = []

        for i, emitter in enumerate(self._emitters):
            emitter_sols = emitter.ask(chat_template=True)
            self._cur_prompts.extend(emitter_sols)
            self._num_emitted[i] = len(emitter_sols)

        batched_results = []
        if len(self._cur_prompts) != 0:
            # check the type of the prompts: If it is a list, then the messages was provided directly.
            # Otherwise, the language prompt was provided directly.
            if not type(self._cur_prompts[0]) == str:
                batched_results = self._model.batch_query_messages(
                    self._cur_prompts,
                    temp=1.1
                )
            else:
                # batch query the results from the provided solutions
                context_batch = np.array([""] * len(self._cur_prompts), dtype=self.archive.dtypes["solution"])

                # batch inference with chat templating
                batched_results = self._model.batch_query(context_batch, self._cur_prompts, temp=1.1, chat=True)

        all_solns = []

        completed = 0
        for i, emitter in enumerate(self._emitters):
            # get the emitter's specific outputs
            e_outputs = batched_results[completed:(completed + self._num_emitted[i])]

            # if the emitter has a post process, process them
            if hasattr(emitter, "post_process"):
                e_outputs = emitter.post_process(e_outputs, model=self._model)
            
            all_solns.extend(e_outputs)

            completed += self._num_emitted[i]

        self._cur_solutions = np.array(all_solns, dtype=self.archive.dtypes["solution"]).reshape(-1, self._solution_dim)

        return self._cur_solutions