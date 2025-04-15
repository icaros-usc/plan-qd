"""Custom GridArchive to allow recording history."""

import ribs.archives

# import OmegaConf


class GridArchive(ribs.archives.GridArchive):
    """Based on pyribs GridArchive.

    This archive records history of its objectives and measure values if
    record_history is True. Before each generation, call new_history_gen() to
    start recording history for that gen. new_history_gen() must be called
    before calling add() for the first time.

    Args:
        *args - Args of pyribs GridArchive.
        **kwargs - Kwargs of pyribs GridArchive.
        record_history - True if history should be recorded. Only saves objective and
            measures by default. (default: True)
        record_solution - True if solution should be included in the history (ignored if
            record_history is True). (default: False)
        record_metadata - True if metadata should be included in the history (ignored if
            record_history is True). (default: False)
    """

    def __init__(
        self,
        *args,
        record_history: bool = True,
        record_solution: bool = False,
        record_metadata: bool = False,
        **kwargs
    ):
        # convert OmegaConf to dict
        kwargs["dtype"] = dict(kwargs["dtype"])

        super().__init__(*args, **kwargs)
        self._record_history = record_history
        self._record_solution = record_solution
        self._record_metadata = record_metadata
        self._history = [] if self._record_history else None

    def new_history_gen(self):
        """Starts a new generation in the history."""
        if self._record_history:
            self._history.append([])

    def history(self):
        """Gets the current history."""
        return self._history

    def add_single(self, solution, objective, measures, **fields):
        add_info = super().add_single(solution, objective, measures, **fields)

        # Save objective and measures (and optionally solutions and metadata) in history,
        # even if not inserted into the archive.
        if self._record_history:
            add_list = ["add_single", objective, measures]
            if self._record_solution:
                add_list += [solution]
            if self._record_metadata:
                add_list += [fields]
            self._history[-1].append(add_list)

        return add_info

    def add(self, solution, objective, measures, **fields):
        add_info = super().add(solution, objective, measures, **fields)

        if self._record_history:
            add_list = ["add", objective, measures]
            if self._record_solution:
                add_list += [solution]
            if self._record_metadata:
                add_list += [fields]
            self._history[-1].append(add_list)

        return add_info
