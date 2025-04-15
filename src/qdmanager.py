"""Manager class that runs the full experiment."""

import dataclasses
import itertools
import logging
import os
import subprocess as sp
import pickle
import uuid

import cloudpickle
import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import yaml

from src.utils.metric_logger import MetricLogger

logger = logging.getLogger(__name__)

class Manager:
    def __init__(self, seed: int, cfg: DictConfig) -> None:
        self._logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        self._cfg = cfg
        self.max_evals = self._cfg["max_evals"]
        self.debug = self._cfg["debug"]
        self.render = self._cfg["render"]
        self.communication = self._cfg["communication"]
        
        try:
            logger.info(f"Attempting to load from {self._logdir}/reload.pkl")

            with open(f"{self._logdir}/reload.pkl", "rb") as f:
                logger.info(f"Reloading experiment from {self._logdir}")
                data = pickle.load(f)
                self.rng = data["rng"]
                self.itrs_completed = data["itrs_completed"]
                self.evals_used = data["evals_used"]
                self.overall_min_obj = data["overall_min_obj"]
                self.metadata_id = data["metadata_id"]
                self.cur_best_id = data["cur_best_id"]
                self.failed_levels = data["failed_levels"]

                if self._cfg["wandb"]["enable"]:
                    if "wandb_id" in data:
                        self.wandb_id = data["wandb_id"]
                    else:
                        logger.warning(
                            "No wandb_id stored in the checkpoint. This will lead to "
                            "creating a new wandb run instead of resuming the old one."
                        )
                        self.wandb_id = str(uuid.uuid4())

                # load the model for the scheduler
                self._setup_agent_model(data["model_cfg"])

                # create the model for the scheduler
                self.scheduler = data["scheduler"]
                self.scheduler.set_model(instantiate(cfg["qd"]["scheduler"]["model"]))

                self.archive = self.scheduler.archive
                if self.scheduler.result_archive is not self.archive:
                    self.result_archive = self.scheduler.result_archive
                else:
                    self.result_archive = None

                self.metrics = data["metrics"]

            self.measure_names = data["measure_names"]

            # Instantiate the env manager
            self.env_manager = instantiate(cfg["env"], measure_names=self.measure_names, rng=self.rng, communication=self.communication)

            logger.info(f"Itrs already completed: {self.itrs_completed}")
            logger.info(
                f"Execution continues from itr {self.itrs_completed + 1} (1-based)"
            )

        except FileNotFoundError:
            logger.info(f"Creating a new experiment at {self._logdir}")
            self.rng = np.random.default_rng(seed=seed)
            self.itrs_completed = 0
            self.evals_used = 0
            self.overall_min_obj = np.inf
            self.metadata_id = 0
            self.cur_best_id = None  # ID of most recent best solution.
            self.failed_levels = []

            # Setup QD components
            self._create_scheduler(seed, cfg)

            # Instantiate the env manager
            self.env_manager = instantiate(cfg["env"], measure_names=self.measure_names, rng=self.rng, communication=self.communication)

            # Setup metrics
            self._setup_metrics()

            # Setup agent model
            self._setup_agent_model(cfg["model"])

            if self._cfg["wandb"]["enable"]:
                self.wandb_id = str(uuid.uuid4())

        # Setup wandb if required
        self.wandb_logger = None
        if self._cfg["wandb"]["enable"]:
            runtime_choices = hydra.core.hydra_config.HydraConfig.get().runtime.choices
            agent_names = [
                a["type"]["_target_"].split(".")[-1] for a in self._cfg["agents"]
            ]
            qd_alg_name = runtime_choices["manager/qd"]
            env_name = runtime_choices["manager/env"]
            tags = agent_names + [qd_alg_name, env_name] + self._cfg["wandb"]["tags"]
            self.wandb_logger = wandb.init(
                project=self._cfg["wandb"]["project_name"],
                entity=self._cfg["wandb"]["entity"],
                config=OmegaConf.to_container(
                    self._cfg, resolve=True, throw_on_missing=True
                ),
                name=f"{', '.join(agent_names)}, {qd_alg_name}, {env_name}",
                id=self.wandb_id,
                tags=tags,
                resume="allow",  # Allow resuming from same run ID
            )

    def _create_scheduler(self, seed, cfg):
        """Instantiate archive and emitters and then the schedulers."""
        logger.info("Setting up scheduler for QD")

        # Get initial solutions
        x0 = cfg["qd"]["x0"]
        mutation_cfg = cfg["qd"]["mutation_cfg"]

        # scale thet solution size by the parts of the prompt being mutated
        num_llm_agents = 0
        for agent in cfg["agents"]:
            if "llm" in agent["type"]['_target_']:
                num_llm_agents += 1

        sol_size = num_llm_agents * len(mutation_cfg["mutation_type"])

        logger.info(f"Solution size: {sol_size}")

        # construct the ranges and 
        dims = [] # : [11, 11]
        ranges = [] # : [[-5, 5], [-5, 5]]
        measure_names = [] # : ["score", "score"]

        for m in cfg["qd"]["measures"]:
            dims.append(m["dim"])
            ranges.append(m["range"])

            measure_names.append(m["name"])

        self.measure_names = measure_names

        # create the archive
        self.archive = instantiate(
            cfg["qd"]["archive"],
            dims=dims,
            ranges=ranges,
            solution_dim=sol_size,
            seed=seed,
            extra_fields={
                "metadata": ((), object),
            },
        )

        # create result archive
        self.result_archive = None
        if "result_archive" in cfg["qd"] and cfg["qd"]["result_archive"] is not None:
            self.result_archive = instantiate(
                cfg["qd"]["result_archive"],
                dims=dims,
                ranges=ranges,
                solution_dim=sol_size,
                seed=seed,
                extra_fields={
                    "metadata": ((), object),
                },
            )
        else:
            self.result_archive = None
        logger.info(f"Archive: {self.archive}")
        logger.info(f"Result archive: {self.result_archive}")

        # Instantiate emitters
        # NOTE: initial solutions are defined in the config file. Specification allows for multiple initial solutions
        # Each emitter has specific config that determines how the initial solutions are used
        emitters = []
        for e in cfg["qd"]["emitters"]:
            emitter_seeds = self.rng.integers(
                np.iinfo(np.int32).max / 2,
                size=e["num"],
                endpoint=True,
            )

            e_kwargs = {
                "archive": self.archive,
                "result_archive": self.result_archive,
                "solution_dim": sol_size,
                "measure_names": list(cfg["qd"]["measures"]),
                "initial_solutions": x0,
                "batch_size": e["batch_size"],
                "prompt": dict(cfg["env"]["module_cfg"]["prompt"]),
                "grid_config": dict(cfg["env"]["module_cfg"]["grid"])
            }

            emitters.extend(
                [
                    instantiate(
                        e["type"],
                        **e_kwargs,
                        **mutation_cfg,
                        seed=s,
                    )
                    for s in emitter_seeds
                ]
            )
        logger.info(f"Emitters: {emitters}")

        # Create custom scheduler
        self.scheduler = instantiate(
            cfg["qd"]["scheduler"],
            archive=self.archive,
            emitters=emitters,
            result_archive=self.result_archive,
        )
        logger.info(f"Scheduler: {self.scheduler}")

    def _setup_metrics(self):
        """Setup metric list for logging."""
        metric_list = [
            ("Total Evals", True),
            ("Mean Evaluation", False),
            ("Actual QD Score", True),
            ("Archive Size", True),
            ("Archive Coverage", True),
            ("Best Objective", False),
            ("Worst Objective", False),
            ("Mean Objective", False),
            ("Overall Min Objective", False),
        ]

        # Extra logging for result archive
        if self.result_archive is not None:
            res_metric_list = [(f"(res) {k}", v) for k, v in metric_list]
            metric_list += res_metric_list

        self.metrics = MetricLogger(metric_list)

    def _setup_agent_model(self, model_cfg):
        """Setup the agent model for the experiment."""

        # we create the agent model here centrally
        self.model = instantiate(model_cfg)

    def _finished(self) -> bool:
        """Whether execution is done."""
        return self.evals_used >= self._cfg["max_evals"]

    def _msg_all(self, msg: str):
        """Logs msg on master, on all workers, and in dashboard_status.txt."""
        logger.info(msg)
        with open(f"{self._logdir}/dashboard_status.txt", "w") as f:
            f.write(msg)

    def _extract_metadata(self, r) -> dict:
        """Constructs metadata object from results of an evaluation."""
        meta = dataclasses.asdict(r)

        # Remove unwanted keys.
        none_keys = [key for key in meta if meta[key] is None]
        for key in itertools.chain(none_keys, []):
            try:
                meta.pop(key)
            except KeyError:
                pass

        meta["metadata_id"] = self.metadata_id
        self.metadata_id += 1

        return meta

    def _evaluate_solutions(self, sols):
        """Evaluates a batch of solutions and adds them to the archive."""
        logger.info("Evaluating solutions")

        skipped_sols = 0
        if self.evals_used + len(sols) > self._cfg["max_evals"]:
            remaining_evals = self._cfg["max_evals"] - self.evals_used
            remaining_sols = remaining_evals
            skipped_sols = len(sols) - remaining_sols
            sols = sols[:remaining_sols]
            logger.info(
                f"Unable to evaluate all solutions; will evaluate "
                f"{remaining_sols} instead"
            )

        self.evals_used += len(sols)

        if self.debug:
            logger.info(f"evals_used (old): {self.evals_used - len(sols)}")
            logger.info(f"evals_used (new): {self.evals_used}")

        logger.info("Distributing evaluations")
        results = self.env_manager.evaluate(
            sols,
            self._cfg["agents"],
            self.model,
            self.debug,
            self.render
        )

        logger.info("Adding solutions to the scheduler")
        objs, measures, metadata, success_mask = [], [], [], []

        for sol, r in zip(sols, results):
            if not r.failed:
                obj = r.agg_obj
                meas = r.agg_measures
                meta = self._extract_metadata(r)

                objs.append(obj)
                measures.append(meas)
                metadata.append(meta)
                success_mask.append(True)
            else:
                failed_level_info = self.env_manager.add_failed_info(sol, r)
                self.failed_levels.append(failed_level_info)
                objs.append(np.nan)
                measures.append(np.full(self.archive.measure_dim, np.nan))
                metadata.append(None)
                success_mask.append(False)

        # Tell results to scheduler.
        logger.info(f"Filling in null values for skipped sols: {skipped_sols}")
        for _ in range(skipped_sols):
            objs.append(np.nan)
            measures.append(np.full(self.archive.measure_dim, np.nan))
            metadata.append(None)
            success_mask.append(False)

        self.scheduler.tell(
            objs,
            measures,
            success_mask=success_mask,
            metadata=metadata,
        )

        self.metrics.add("Mean Evaluation", np.nanmean(objs), logger)
        self.overall_min_obj = min(self.overall_min_obj, np.nanmin(objs))

        # reset all modules
        self.env_manager._reset_all_modules(hard=True)

    def _add_performance_metrics(self, archive, prefix="", commit=False):
        """Calculates various performance metrics at the end of each iter."""
        objs = archive.data("objective")
        stats = archive.stats

        metrics_dict = {
            "Total Evals": self.evals_used,
            "Actual QD Score": self.env_manager.actual_qd_score(objs),
            "Archive Size": stats.num_elites,
            "Archive Coverage": stats.coverage,
            "Best Objective": np.max(objs),
            "Worst Objective": np.min(objs),
            "Mean Objective": np.mean(objs),
            "Overall Min Objective": self.overall_min_obj,
        }

        for k, v in metrics_dict.items():
            self.metrics.add(f"{prefix}{k}", v, logger)

        if self.wandb_logger is not None:
            # Plot archive heatmap
            measure_names = self.measure_names
            if len(measure_names) != 2:
                logger.warning(
                    "Number of measures is not 2. Skipping archive heatmap from wandb logs."
                )
            else:
                import matplotlib.pyplot as plt
                from ribs.visualize import grid_archive_heatmap

                fig, ax = plt.subplots(1, 1)
                grid_archive_heatmap(
                    archive,
                    ax=ax,
                    vmin=self.env_manager.module.MIN_SCORE,
                    vmax=self.env_manager.module.MAX_SCORE,
                    cmap="viridis",
                    rasterized=True,
                )
                fig.suptitle("Archive heatmap")
                ax.set_xlabel(measure_names[0])
                ax.set_ylabel(measure_names[1])
                im = wandb.Image(fig)
                metrics_dict["Archive heatmap"] = im
                plt.close(fig)

            # Log to wandb
            wandb_prefix = "Train Archive" if prefix == "" else "Result Archive"
            wandb.log(
                {f"{wandb_prefix}/{k}": v for k, v in metrics_dict.items()},
                commit=commit,
                step=self.itrs_completed,
            )

    def _plot_metrics(self):
        """Plots metrics every plot_metrics_freq itrs or on final itr."""
        if (
            self.itrs_completed % self._cfg["plot_metrics_freq"] == 0
            or self._finished()
        ):
            logger.info(f"Metrics:\n{self.metrics.get_plot_text()}")

    def _save_archive_history(self):
        """Saves the archive's history.

        We are okay with a pickle file here because there are only numpy arrays
        and Python objects, both of which are stable.
        """
        if self.result_archive is not None:
            with open(f"{self._logdir}/result_archive_history.pkl", "wb") as f:
                pickle.dump(self.result_archive.history(), f)
        with open(f"{self._logdir}/archive_history.pkl", "wb") as f:
            pickle.dump(self.archive.history(), f)

    def _save_archive(self):
        """Saves dataframes of the archive.

        The archive, including solutions and metadata, is saved to
        logdir/archive/archive_{itr}.pkl

        Note that the archive is saved as an dict storing common
        Python objects, so it should be stable (at least, given fixed software
        versions).
        """
        os.makedirs(f"{self._logdir}/archive", exist_ok=True)
        itr = self.itrs_completed
        if self.result_archive is not None:
            data = self.result_archive.data()
            with open(f"{self._logdir}/archive/result_archive_{itr}.pkl", "wb") as f:
                pickle.dump(data, f)

        data = self.archive.data()
        with open(f"{self._logdir}/archive/archive_{itr}.pkl", "wb") as f:
            pickle.dump(data, f)

    def _save_reload_data(self):
        """Saves data necessary for a reload.

        Current reload files:
        - reload.pkl

        Since saving may fail due to memory issues, data is first placed in
        reload-tmp.pkl. reload-tmp.pkl then overwrites reload.pkl.
        """

        # for QD-LLM, the scheduler contains contents which cannot be pickled
        scheduler = self.scheduler.serializable_copy()

        logger.info("Saving reload data")
        logger.info("Saving reload-tmp.pkl")
        with open(f"{self._logdir}/reload-tmp.pkl", "wb") as file:
            reload_data = {
                "rng": self.rng,
                "itrs_completed": self.itrs_completed,
                "evals_used": self.evals_used,
                "overall_min_obj": self.overall_min_obj,
                "metadata_id": self.metadata_id,
                "cur_best_id": self.cur_best_id,
                "failed_levels": self.failed_levels,
                "scheduler": scheduler,
                "metrics": self.metrics,
                "model_cfg": self._cfg["model"],
                "measure_names": self.measure_names,
            }

            if self._cfg["wandb"]["enable"]:
                reload_data["wandb_id"] = self.wandb_id

            cloudpickle.dump(reload_data, file)

        logger.info("Renaming tmp reload files")
        os.rename(f"{self._logdir}/reload-tmp.pkl", f"{self._logdir}/reload.pkl")

        logger.info("Finished saving reload data")

    def _save_data(self):
        """Saves archive, reload data, history, and metrics if necessary.

        This method must be called at the _end_ of each itr. Otherwise, some things
        might not be complete. For instance, the metrics may be in the middle of an
        iteration, so when we reload, we get an error because we did not end the
        iteration.
        """
        if self._cfg["archive_save_freq"] is None:
            save_full_archive = False
        elif self._cfg["archive_save_freq"] == -1 and self._finished():
            save_full_archive = True
        elif (
            self._cfg["archive_save_freq"] > 0
            and self.itrs_completed % self._cfg["archive_save_freq"] == 0
        ):
            save_full_archive = True
        else:
            save_full_archive = False

        logger.info("Saving metrics")
        self.metrics.to_json(f"{self._logdir}/metrics.json")

        logger.info("Saving archive history")
        self._save_archive_history()

        if save_full_archive:
            logger.info("Saving full archive")
            self._save_archive()
        if (
            self.itrs_completed % self._cfg["reload_save_freq"] == 0
        ) or self._finished():
            self._save_reload_data()
        if self._finished():
            logger.info("Saving failed levels")
            with open(f"{self._logdir}/failed_levels.pkl", "wb") as f:
                pickle.dump(self.failed_levels, f)

    def run_experiment(self):
        """Runs an experiment."""
        while not self._finished():
            self._msg_all(
                f"----- Itr {self.itrs_completed + 1} "
                f"({self.evals_used} evals) -----"
            )
            self.metrics.start_itr()
            self.archive.new_history_gen()
            if self.result_archive is not None:
                self.result_archive.new_history_gen()

            logger.info("Running QD ask/tell")
            sols = self.scheduler.ask()
            self._evaluate_solutions(sols)

            logger.info("Itr complete - now logging and saving data")
            self.itrs_completed += 1

            # Metrics + optional wandb logging
            if self.result_archive is not None:
                self._add_performance_metrics(self.result_archive, "(res) ")
            self._add_performance_metrics(self.archive, commit=True)
            self.metrics.end_itr()
            self._plot_metrics()
            self._save_data()  # Keep at end of loop (see method docstring).

        self._msg_all(
            f"----- Done! {self.itrs_completed} itrs, " f"{self.evals_used} evals -----"
        )

        if self.wandb_logger is not None:
            self.wandb_logger.finish()
