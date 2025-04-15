import time

import numpy as np
from overcooked_ai_py.mdp.overcooked_env import MAX_HORIZON, OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS

from .steakhouse_mdp import EVENT_TYPES, SteakhouseGridworld


class SteakhouseEnv(OvercookedEnv):
    def __init__(
        self,
        mdp_generator_fn,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        num_mdp=1,
        initial_info={},
    ):
        super().__init__(
            mdp_generator_fn,
            start_state_fn,
            horizon,
            mlam_params,
            info_level,
            num_mdp,
            initial_info,
        )

    @staticmethod
    def from_mdp(
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=1,
        num_mdp=None,
    ):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, SteakhouseGridworld)
        if num_mdp is not None:
            assert num_mdp == 1
        mdp_generator_fn = lambda _ignored: mdp
        return SteakhouseEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1,
        )

    def copy(self):
        # TODO: Add testing for checking that these util methods are up to date?
        return SteakhouseEnv(
            mdp_generator_fn=self.mdp_generator_fn,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            info_level=self.info_level,
            num_mdp=self.num_mdp,
        )

    ###################
    # BASIC ENV LOGIC #
    ###################

    def step(self, joint_action, joint_agent_action_info=None, display_phi=False):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        if joint_agent_action_info is None:
            joint_agent_action_info = [{} for _ in range(self.mdp.num_players)]
        next_state, mdp_infos = self.mdp.get_state_transition(
            self.state, joint_action, display_phi, self.mp
        )

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)

        if done:
            self._add_episode_info(env_info)

        # set start time
        if self.first_action_taken:
            self.start_time = time.time()
            self.first_action_taken = False

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        return (next_state, timestep_sparse_reward, done, env_info)

    def lossless_state_encoding_mdp(self, state):
        """
        Wrapper of the mdp's lossless_encoding
        """
        return self.mdp.lossless_state_encoding(state, self.horizon)

    def featurize_state_mdp(self, state, num_pots=2):
        """
        Wrapper of the mdp's featurize_state
        """
        return self.mdp.featurize_state(state, self.mlam, num_pots=num_pots)

    def reset(self, regen_mdp=True, outside_info={}):
        """
        Resets the environment. Does NOT reset the agent.
        Args:
            regen_mdp (bool): gives the option of not re-generating mdp on the reset,
                                which is particularly helpful with reproducing results on variable mdp
            outside_info (dict): the outside information that will be fed into the scheduling_fn (if used), which will
                                 in turn generate a new set of mdp_params that is used to regenerate mdp.
                                 Please note that, if you intend to use this arguments throughout the run,
                                 you need to have a "initial_info" dictionary with the same keys in the "env_params"
        """
        if regen_mdp:
            self.mdp = self.mdp_generator_fn(outside_info)
            self._mlam = None
            self._mp = None
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
        }
        self.game_stats = {**events_dict, **rewards_dict}
        self.start_time = None
        self.first_action_taken = True

    def is_done(self):
        """Whether the episode is over."""
        return self.state.timestep >= self.horizon or self.mdp.is_terminal(self.state)

    ####################
    # TRAJECTORY LOGIC #
    ####################

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display:
            print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display:
                print(self)
            if done:
                break
        successor_state = self.state
        self.reset(False)
        return successor_state, done

    ##################
    #   RENDERING    #
    ##################

    def render(self, mode="human", chat_messages=None):
        time_step_left = (
            self.horizon - self.state.timestep if self.horizon != MAX_HORIZON else None
        )
        time_passed = (
            time.time() - self.start_time if self.start_time is not None else 0
        )
        self.mdp.render(
            self.state,
            mode,
            time_step_left=time_step_left,
            time_passed=time_passed,
            chat_messages=chat_messages
        )
