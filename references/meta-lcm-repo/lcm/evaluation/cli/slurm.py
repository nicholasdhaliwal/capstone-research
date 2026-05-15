# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import asyncio
import logging
import time

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from stopes.core import Requirements

from lcm.evaluation.arun import EvalRunModule, RunModuleConfig, schedule_task
from lcm.evaluation.cli.configs import CliConfig, LauncherOptions, parse_configs
from lcm.utils.common import promote_config, setup_conf

logger = logging.getLogger("lcm.evaluation")


def main(
    cfg: CliConfig, launcher_opts: LauncherOptions, logger: logging.Logger = logger
) -> None:
    """
    Pipeline main steps:
    - Create multiple EvalModuleConfig for each task
    - Schedule and run the (sharded) tags on SLURM
    - Aggregate the metrics in the scheduler node
    """

    job_args = getattr(cfg, "job_args", None) or {}

    if isinstance(job_args, DictConfig):
        job_args = OmegaConf.to_container(job_args)

    assert isinstance(job_args, dict), f"Unexpected `job_args` type: {type(job_args)}"

    # 1. Set up launcher
    # If launcher_opts is a string (i.e. passed via non-Hydra CLI), we set up an embedded
    # Hydra session to construct the launcher
    if isinstance(launcher_opts, DictConfig):
        launcher = hydra.utils.instantiate(launcher_opts)
    else:
        launcher_args = []
        for k, v in job_args.items():  # type: ignore
            if k.startswith("launcher."):
                # Escape list-style string for Hydra
                if v and "," in v:
                    launcher_args.append(f"++{k}='{v}'")
                else:
                    launcher_args.append(f"++{k}={v}")

        setup_conf()  # Register stopes' and lcm' launchers
        with initialize(version_base="1.2", config_path="../../../recipes/common"):
            launcher_cfg = compose(
                config_name="requirements",  # load any config to attach and detach launcher later
                overrides=[f"+launcher={launcher_opts}"] + launcher_args,
            )

        launcher = hydra.utils.instantiate(launcher_cfg)["launcher"]

    # 2. Set up requirements
    requirements_args = job_args.get("requirements", None)
    if requirements_args:
        requirements = promote_config(requirements_args, Requirements)
    else:
        requirements = None
    nshards = job_args.get("nshards", None)  # type: ignore

    # 3. Set up run
    run_configs = parse_configs(cfg)
    task_names = [r.task_name for r in run_configs]
    task_modules = [
        EvalRunModule(
            RunModuleConfig(
                requirements=requirements, nshards=nshards, **run_config.__dict__
            )
        )
        for run_config in run_configs
    ]
    start = time.monotonic()
    loop = asyncio.get_event_loop()
    all_runs = asyncio.gather(
        *[schedule_task(m, launcher, logger=logger) for m in task_modules]
    )
    loop.run_until_complete(all_runs)
    logger.info(
        f"Tasks {task_names} took {time.monotonic() - start:.2f} seconds (including scheduling)."
    )
