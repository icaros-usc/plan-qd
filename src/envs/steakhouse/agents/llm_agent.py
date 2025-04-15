"""LLM agent file"""

import itertools
import os
import sys
from typing import Dict

import numpy as np
from overcooked_ai_py.mdp.actions import Action

from .qd_agent import QDAgent

class LLMAgent(QDAgent):
    """
    this is LLM model execute subtasks in advanced steakhouse environment
    """

    def __init__(
        self,
        mlam_config,
        prompt: Dict[str, str],
        memory_depth=0,
        prompt_overrides: Dict[str, str] = {},
        auto_unstuck: str = "none",
        **kwargs
    ):
        self.reset()

        self.prompt = prompt
        self.mlam_config = mlam_config

        self.memory_depth = memory_depth

        # prompt overrides contain the mutated parts of the prompt
        self.prompt_overrides = prompt_overrides

        # below are greedy human model parameters

        # Bool for perfect rationality vs Boltzmann rationality for high level
        # and low level action selection
        self.hl_boltzmann_rational = (
            False  # For choices among high level goals of same type
        )

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = 1
        self.ll_temperature = 1

        # Whether to automatically take an action to get the agent unstuck if
        # it's in the same state as the previous turn. If "none", the agent is
        # history-less, while if "original" it has history.
        self.auto_unstuck = auto_unstuck

        self.current_ml_action = 0
        
        self.curr_obj = None
        self.prev_obj = None
        
        self.task_list = None

    def set_prompt_overrides(
        self,
        prompt_overrides: Dict[str, str] = {},
    ):
        """Set the prompt overrides for the agent"""
        self.prompt_overrides = prompt_overrides

    def set_model(self, model):
        """Set the LLM model for the agent to the provided model. This
        is passed as a setter as opposed to a parameter to avoid reconstructing
        the same model for each agent.
        """
        self.model = model

    def set_agent_index(self, agent_index):
        return super().set_agent_index(agent_index)

    def get_context(self):
        """Get the context of the agent"""
        return self.prompt_overrides["context"]

    def set_mdp_and_mlam(self, mdp, mlam, **kwargs):
        super().set_mdp(mdp)

        self.prompt_builder = kwargs["prompt_builder"]
        self.mlam = mlam

    def ml_action(self, state):
        return self.motion_goals

    def action(self, state, timestep=None):
        # append action name to action names list
        self.ml_action_log.append(self.current_ml_action)

        # ml_action is determined by llm
        possible_motion_goals = self.ml_action(state)

        # enforce the agent to stay if there are no possible motion goals
        if possible_motion_goals == Action.STAY:
            chosen_goal = possible_motion_goals
            chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)
        else:
            # this implies that the agent is moving
            if timestep is not None:
                self.last_update_timestep = timestep

            # Once we have identified the motion goals for the medium
            # level action we want to perform, select the one with lowest cost
            start_pos_and_or = state.players_pos_and_or[self.agent_index]

            chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
                start_pos_and_or, possible_motion_goals
            )

        if self.auto_unstuck == "original":
            # HACK: if agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if (
                self.prev_state is not None
                and state.players_pos_and_or == self.prev_state.players_pos_and_or
            ):
                action_lists = [[Action.STAY]] * self.mdp.num_players
                action_lists[self.agent_index] = Action.ALL_ACTIONS
                joint_actions = list(itertools.product(*action_lists))

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
                    if new_state.player_positions != self.prev_state.player_positions:
                        unblocking_joint_actions.append(j_a)

                # Add an option for STAY as well since the other agents can actually
                # move and STAY could be the best way to resolve collision. The above if
                # statement will not include STAY and INTERACT since they won't change
                # the player positions. INTERACT is not added since a random INTERACT
                # might mess things up by placing or picking things unintentionally.
                unblocking_joint_actions.append([Action.STAY] * self.mdp.num_players)
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state

        return chosen_action, {"action_probs": action_probs}

    def get_raw_data(self):
        """Return the agent's raw data"""
        return {"agent_index": self.agent_index, "ml_action_log": self.ml_action_log}

    def build_update_prompt(
        self,
        state,
        prompt_format = None,
        communication_enabled: bool = True,
        explicit_language: bool = True,
        **kwargs
    ):
        """Build the update prompt that the agent will use to update its medium level action
        
        Provide a prompt_format to override the default prompt format provided
        """
        cross_reference, modified_prompt_overrides, task_list = self.prompt_builder.build(
            state,
            self.mlam,
            self.agent_index,
            self.prompt,
            self.prompt_overrides,
            self.ml_action_log,
            self.memory_depth,
            communication_enabled=communication_enabled,
            explicit_language=explicit_language,
            **kwargs
        )
        self.task_list = task_list

        if prompt_format is None:
            prompt_format = self.prompt["messages"]

        messages = []
        for struct in prompt_format:
            role = struct[0]
            content = struct[1]

            new_content = []
            for entry in content:
                if entry[0] == "text":
                    new_content.append(entry[1])
                else:
                    # order of precedence:
                    # 1. references (contains mutated parts of the prompt)
                    # 2. regular prompt
                    new_entry = modified_prompt_overrides.get(
                        entry[1], None
                    ) or self.prompt.get(
                        entry[1], None
                    )  # or self.kwargs.get(entry[1], None)
                    if new_entry:
                        new_content.append(new_entry)

            final_string = "\n".join(new_content)
            messages.append({"role": role, "content": final_string})

        # return the cross reference and corresponding prompt
        return cross_reference, messages
    
    def update_chat_message(self, idx):
        """Update the current objective for the chat message"""
        self.prev_obj = self.curr_obj
        self.curr_obj = self.task_list[idx]
        

    def update_motion_goals(self, state, xref, timestep=None):
        """Update motion goals from ml action"""

        # log the last update timestep
        if timestep is not None:
            self.last_update_timestep = timestep

        # log subtask index in agent history
        self.current_ml_action = xref[1]

        # obtain relevant state information
        player = state.players[self.agent_index]
        am = self.mlam

        # the subtask_index is already the motion goals for the agent
        motion_goals = xref[0]

        if motion_goals == Action.STAY:
            self.motion_goals = Action.STAY
            return

        # filter out invalid motion goals
        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        # if no valid motion goals, go to the closest feature
        if motion_goals == []:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [
                mg
                for mg in motion_goals
                if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                    player.pos_and_or, mg
                )
            ]

        # update the motion goals
        self.motion_goals = motion_goals

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.

        Adapted from the original OvercookedAI GreedyHumanModel.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [
                self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
                for goal in motion_goals
            ]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(
                plan_costs, self.hl_temperature
            )
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.

        Adapted from the original OvercookedAI GreedyHumanModel.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def reset(self):
        """Reset the agent's memory"""
        super().reset()

        # reset local metrics
        self.prev_state = None
        self.last_update_timestep = 0

        # clear other agent metrics
        self.ml_action_log = []  # store low level actions
        self.other_agent_actions = []
        self.motion_goals = []

    @property
    def initial_solution(self):
        return self.personality
