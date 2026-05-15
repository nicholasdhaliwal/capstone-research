# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
import time

import torch

from lcm.evaluation.cli.configs import CliConfig, parse_configs
from lcm.evaluation.run import run_task
from lcm.evaluation.utils.common import (
    flatten_dict,
    format_dict,
    log_final_results,
    write_to_json,
)
from lcm.evaluation.utils.distributed import get_gang, get_global_rank, rank_zero_info

logger = logging.getLogger("lcm.evaluation")


def main(cfg: CliConfig, logger: logging.Logger = logger) -> None:
    run_configs = parse_configs(cfg)

    assert len(run_configs) > 0, f"No tasks were found given pattern '{cfg.tasks}'"
    run_ids = [r.name for r in run_configs]
    rank_zero_info(f"Selected task execution: {run_ids}")

    all_runs_metrics = {}

    for run_config in run_configs:
        name = run_config.name
        rank_zero_info(f"Running evaluation on task {name}", logger=logger)
        start = time.monotonic()

        metrics, result_file = run_task(run_config, logger=logger, gang=get_gang())
        if run_config.dump_dir is not None and get_global_rank() == 0:
            result_content = {
                "results": flatten_dict(metrics),
                "configs": run_config.params,
            }
            rank_zero_info(f"Writing metric results to {result_file}", logger=logger)
            write_to_json(result_content, result_file, indent=4)

        log = format_dict(flatten_dict(metrics), delimiter=" | ", decimal=6)
        rank_zero_info(f"Evaluation results on task {name}: {log}", logger=logger)
        rank_zero_info(
            f"Task {name} took {time.monotonic() - start:.2f} seconds", logger=logger
        )
        all_runs_metrics[name] = metrics
        torch.cuda.empty_cache()

    results = flatten_dict(all_runs_metrics)
    rank_zero_info(f"All evaluation results: {format_dict(results)}", logger=logger)
    log_final_results(
        results, cfg.predictor_config, cfg.tb_log_dir, cfg.metric_log_dir, logger
    )
