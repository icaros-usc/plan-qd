"""Provides QDAgent."""

import numpy as np
from overcooked_ai_py.agents.agent import Agent


class QDAgent(Agent):
    """Base class for agents for QD search."""

    def from_numpy(self, params: np.ndarray) -> "self":
        """Set agent parameters from a 1D numpy array.

        Args:
            params: Values of the params to use.
        Returns:
            Returns self (literally, "return self") so that it can be chained with other
            methods.
        """
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        """Convert agent parameters to a 1D numpy array.

        Returns:
            1D array of current parameter values.
        """
        raise NotImplementedError

    @property
    def num_parameters(self):
        """Return the number of trainable parameters required for this agent."""
        raise NotImplementedError

    @property
    def initial_solution(self):
        return np.zeros(self.num_parameters)
