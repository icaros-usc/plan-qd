"""Provides SteakhouseModule."""

import logging
from typing import Dict, List, Optional

import numpy as np
from collections import deque
import time
from overcooked_ai_py.agents.agent import Agent

from .mdp.steakhouse_env import SteakhouseEnv
from .mdp.steakhouse_mdp import SteakhouseGridworld, SteakhouseState
from .planning.steak_planner import SteakMediumLevelActionManager

from src.prompt.utils import Prompt

import random

logger = logging.getLogger(__name__)


class SteakhouseModule:
    """Module for Steakhouse.

    Args:
        grid: List of list of characters representing the level.
        grid_world_config: Config for the grid world.
        mlam_config: Config for SteakMediumLevelActionManager that is used to
            determine low-level actions from medium-level actions.
        horizon: Time horizon for the environment.
        kwargs: Only for backward compatibility with removed args.
    """

    def __init__(
        self,
        grid: Dict[str, List[str]],
        grid_world_config: dict,
        horizon: int,
        mlam_config: dict = None,
        prompt_builder: Optional[Prompt] = None,
        max_requery_time: int = 50,
        all_update_on_interact: bool = True,
        chat_history_length: int = 1,
        discount_factor: float = 0.999,
        communication: Dict[str, str] = {},
        explicit_language: bool = False,
        **kwargs,
    ):
        self.grid_config = grid
        self.grid = grid['grid']
        self.grid_language_description = grid['description']
        self.shared_counter_locations = grid.get("shared_counter_locations", [])

        self.grid_world_config = grid_world_config
        self.prompt_builder = prompt_builder
        self.max_requery_time = max_requery_time

        mlam_params = {
            "start_orientations": False,
            "wait_allowed": False,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
        }

        self.mlam_config = mlam_config or mlam_params
        self.horizon = horizon

        self.all_update_on_interact = all_update_on_interact
        self.chat_history_length = chat_history_length
        self.init_chat_history_length = chat_history_length

        self.communication = communication
        self.communication_enabled = communication.get("enabled", True)

        self.explicit_language = explicit_language

        self.total = []

        # discounting for sparse reward
        self.step_discount = 1
        self.discount_factor = discount_factor

        # query queue and other variables are set in _reset_metrics
        self._setup()

    def maybe_render(
        self,
        render = False,
        img_name = None
    ):
        """Renders the env and saves an img of it if needed."""
        if render:
            # get chat messages
            chat_messages = [chat for chat in self.chat_history[-6:] if chat[1] == self.timestep]
            self.env.render(
                chat_messages=chat_messages
            )
            # if img_name is not None:
            #     cur_name = img_name(self.timestep)
            #     pygame.image.save(self.env.mdp.viewer, cur_name)
            time.sleep(0.5)

    def inc_queries(self, num: int = 1):
        """Increments the number of queries"""
        self.queries += num

    def _step(
        self,
        joint_action: List,
        render: Optional[bool] = False,
        img_name: Optional[callable] = None
    ):
        """Steps the environment."""
        joint_agent_action_info = []
        for i in range(len(joint_action)):
            joint_agent_action_info.append({str(i)})

        next_state, timestep_sparse_reward, done, info = self.env.step(
            tuple(joint_action), # format the joint action as a tuple
            joint_agent_action_info=joint_agent_action_info
        )

        # render the environment if needed
        self.maybe_render(
            render = render,
            img_name = img_name
        )

        # log the joint action
        self.joint_actions.append(joint_action)
            
        # log metrics
        self.total_sparse_reward += timestep_sparse_reward * self.step_discount
        self.step_discount *= self.discount_factor

        self.total_fitness += timestep_sparse_reward

        self.done = done
        self.last_state = next_state
        self.state_history.append(next_state.to_dict(all=False))

        if timestep_sparse_reward > 0:
            self.deliver_history.append([self.timestep, timestep_sparse_reward])

    def _step_and_get_agents(
        self,
        agent_list: List[Agent],
        render: Optional[bool] = False,
        img_name: Optional[callable] = None
    ):
        """Creates the joint action of all agents and steps the environment. Returns a List of agents
        that need to be updated. Note that the first agent in the agents_to_update list is the agent that
        initiated the query.

        NOTE: timestep is NOT incremented here. It is incremented in the main loop of the respective communication style.
        """

        joint_action = []
        agents_to_update = []

        for agent in agent_list:
            agent_action = agent.action(self.env.state, self.timestep)
            joint_action.append(agent_action[0])

            # check for prompt updates below: only for LLM agents
            if type(agent).__name__ != "LLMAgent":
                continue

            should_update = False

            if agent_action[0] == "interact" or self.timestep - agent.last_update_timestep > self.max_requery_time:
                # if enabled, all agents will be updated on interact
                if self.all_update_on_interact:
                    # add agent to update list
                    agents_to_update.clear()
                    agents_to_update.append(agent)

                    # add all other LLM agents to the update list
                    for agent2 in agent_list:
                        # only add LLM agents to the update list
                        if type(agent2).__name__ == "LLMAgent" and agent2 != agent:
                            agents_to_update.append(agent2)
                else:
                    should_update = True

            # make sure we should update and the agent list isn't already full (aka all agents are already in the list)
            if should_update and not (len(agents_to_update) == len(agent_list)):
                agents_to_update.append(agent)

        self._step(joint_action, render, img_name)
    
        # if done, append workloads to the player workload list
        if self.done:
            # add workloads to the player workload list
            workloads = self.last_state.get_player_workload()
            self.player_workload_list.extend(workloads)

        # return the agents to update
        return agents_to_update

    def _get_language_agents(
        self,
        agent_list: List[Agent]
    ):
        """Extracts only the language agents from the provided list
        """
        new_list = []
        for agent in agent_list:
            # only add LLM agents to the update list
            if type(agent).__name__ == "LLMAgent":
                new_list.append(agent)

        return new_list

    def _populate_query_queue(
        self,
        agent_indices: List[int],
        random:bool=True,
        blacklist: Optional[List[int]] = None
    ):
        if random: # shuffle the list
            np.random.shuffle(agent_indices)

        for agent in agent_indices:
            if blacklist is not None and agent in blacklist:
                continue
            self.query_queue.append(agent.agent_index)

    def step_and_log_metrics(
        self,
        agent_list: List[Agent],
        render: Optional[bool] = False,
        img_name: Optional[callable] = None,
    ):
        """Obtains joint actions and steps the environment. Logs
        metrics for evaluation.

        Returns
            prompts_batch: batch of LLM query prompts to be processed in batch.
            These prompts are all processed in the manager, and returned to the
            module for updating the relevant agents.
        """
        prompts_batch = []
        xref_batch = []
        agents_to_update = []

        # randomly starts any communication sequence with an agent from the list
        random_start = self.communication.get("random_start", True)

        # if sequential, this would be back and forth style communication
        if self.communication.get("sequential", False):
            # if timestep = 0, we populate the queue with all agents
            if not self.already_updated and self.timestep == 0 and len(self.query_queue) == 0:
                # only add LLM agents to the update queue
                new_list = self._get_language_agents(agent_list)
                self._populate_query_queue(new_list, random_start)

            if len(self.query_queue) > 0:
                # if the queue is empty, push the timestep

                # pop the first agent number
                agent_num = self.query_queue.popleft()
                agent = agent_list[agent_num]

                # now, we update the agent with the current state

                # get the past `chat_history_length` messages from the chat history
                messages = self.chat_history[-self.chat_history_length:]
                b_kwargs = {
                    "chat_history": messages,
                    "agent_types": [type(a).__name__ for a in agent_list],
                    "communication_enabled": self.communication_enabled,
                    "explicit_language": self.explicit_language,
                    "horizon": self.horizon,
                }

                # obtain the formatted prompt and xref from the agent
                xref, prompt = agent.build_update_prompt(self.env.state, **b_kwargs)
                xref_batch.append(xref)
                prompts_batch.append(prompt)
                agents_to_update.append(agent)

                self.already_updated = True
            else:
                # step the environment, and see if we need to update the agents
                agents_to_queue = self._step_and_get_agents(agent_list, render, img_name)
                self._populate_query_queue(agents_to_queue, random=random_start)

                # push timestep
                self.timestep += 1

            self.agents_to_update = agents_to_update
            self.xref_list = xref_batch

            return prompts_batch

        # if done, do not update the agents: return an empty list
        return []

    def _parse_response(
        self,
        response: str
    ):
        """Export metrics from the agent response

        TODO: implement more robust parsing/prompting method
        """
        # extract the metrics from the prompt
        message_to_other_chef = None
        subtask_index = 0

        # resolve message to other chef
        if response.find("[") != -1:
            message_to_other_chef = response[
                response.find("[") + 1 : response.find("]")
            ]

            if message_to_other_chef == "":
                message_to_other_chef = None

        # parse response for subtask index and cross reference index to subtasks list
        ind = str.rfind(response, "Option ")
        try:
            subtask_index = int(response[ind + 7 : ind + 8])
        except Exception as e:
            logger.warning("Could not find response when parsing")
            subtask_index = 0
        
        return {
            "message": message_to_other_chef,
            "ml_action": subtask_index
        }

    def update_agents_ml_action(
        self,
        responses: list
    ):
        """Update all agents with the current state."""
        state = self.env.state

        for agent, xref, response in zip(self.agents_to_update, self.xref_list, responses):
            # add agent responses to the response history
            # we subtract 1 from the timestep to get the correct timestep for the response
            self.response_history.append([agent.agent_index, self.timestep, response])

            try: 
                # parse contents and extract relevant information
                contents = self._parse_response(response)

                # override to stay if the provided action is out of bounds
                ml_action = contents.get("ml_action", 0)
                if ml_action < 0 or ml_action >= len(xref):
                    ml_action = 0

                if self.communication_enabled:
                    agent.update_chat_message(ml_action - 1)
                    message = contents.get("message", None)
                else:
                    message = None

                # xref to correct ML action and update agent
                xref_ml_action = xref[ml_action - 1]
                agent.update_motion_goals(state, xref_ml_action, timestep = (self.timestep))

                if message is not None:
                    # add to chat history
                    self.chat_history.append([agent.agent_index, self.timestep, message])
            except Exception as e:
                logger.error(f"Error updating agent {agent.agent_index} with action {ml_action - 1} in list {xref}: {e}")
                logger.error(f"Agent response: {response}")
                
                # print trace
                import traceback
                traceback.print_exc()

                # since the query failed, ensure that the agent is updated next timestep: we set their last update timestep to beyond the requery time
                agent.last_update_timestep = -self.max_requery_time

    def get_raw_data(
        self,
        agents: List[Agent]
    ):
        """Compile raw metadata from the evaluation. Metadata is aggregated
        in the SteakhouseManager.
        """
        raw_data = {
            "fitness_list": self.total_fitness,
            "total_sparse_reward_list": self.total_sparse_reward,
            "joint_actions_list": self.joint_actions,
            "chat_history": self.chat_history,
            "deliver_history": self.deliver_history,
            "response_history": self.response_history,
            "agent_data": [a.get_raw_data() for a in agents],
            "state_history": self.state_history,
            "player_workload_list": self.player_workload_list,
            "queries": self.queries
        }

        return raw_data

    def _setup(self):
        """Set up Steakhouse mdp and env for evaluation"""
        # create MDP and env
        misc = {
            "shared_counter_locations": self.shared_counter_locations
        }
        self.mdp = SteakhouseGridworld.from_grid(
            self.grid,
            self.grid_world_config,
            explicit_language=self.explicit_language,
            misc=misc
        )
        self.env = SteakhouseEnv.from_mdp(self.mdp, info_level=0, horizon=self.horizon)

        FULL_PARAMS = {
            "start_orientations": False,
            "wait_allowed": False,
            "counter_goals": self.mdp.terrain_pos_dict["X"],
            "counter_drop": self.mdp.terrain_pos_dict["X"],
            "counter_pickup": self.mdp.terrain_pos_dict["X"],
            "same_motion_goals": False,
        }

        # create MLAM
        self.mlam = SteakMediumLevelActionManager(self.mdp, FULL_PARAMS)

        # create prompt builder
        grid_layout = self.mdp.terrain_mtx
        self.prompt_builder.set_grid_layout(grid_layout)
        self.prompt_builder.set_grid_config(self.grid_config)
        # self.prompt_builder.set_grid_grid_language_description(self.grid_language_description)

        self._reset_metrics(hard=True)

    def _reset_metrics(self, hard=False):
        """Set up metrics for evaluation."""
        
        self.done = False
        self.total_sparse_reward = 0
        self.total_fitness = 0
        self.last_state = self.env.state
        self.timestep = 0

        # reset the query queue
        self.query_queue = deque()

        # reset the module variables
        self.already_updated = False

        self.xref_list = []
        self.agents_to_update = []

        self.joint_actions = []
        self.chat_history = []
        self.deliver_history = []
        self.response_history = []
        self.player_workload_list = []
        
        self.queries = 0

        # internal representation of the state history. populate initially.
        # with the first state of the environment
        self.state_history = [self.last_state.to_dict(all=False)]

    def _reset_env(self, hard=False):
        """Re-creates an SteakhouseEnv from the MDP."""
        self.env = SteakhouseEnv.from_mdp(self.mdp, info_level=0, horizon=self.horizon)

        self.step_discount = 1
        self.comm_counter = 0
        self.total = []
        
        # reset the query queue
        self.query_queue = deque()

        self._reset_metrics(hard=hard)
