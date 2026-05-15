# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import sys
import typing as tp

from lcm.evaluation.cli.configs import CliConfig, LauncherOptions
from lcm.evaluation.cli.params import (
    extract_args_from_cli,
    from_cli,
    parse_args,
    to_cli,
)
from lcm.evaluation.predictors import get_config_cls
from lcm.evaluation.utils.common import initialize_logger


def cfg_from_cli(
    args: tp.Optional[tp.Sequence[str]] = None,
) -> tp.Tuple[CliConfig, tp.Optional[LauncherOptions]]:
    known_eval_cfg, unknown = to_cli(CliConfig).parse_known_args(args)
    cfg: CliConfig = from_cli(CliConfig, vars(known_eval_cfg), allow_incomplete=True)

    # Extract data configs (dataset and data_loading)
    dataset_args, unknown = extract_args_from_cli(unknown, prefix="dataset.")
    if dataset_args:
        cfg.dataset_args = dataset_args
    dataloading_args, unknown = extract_args_from_cli(unknown, prefix="data_loading.")
    if dataloading_args:
        cfg.dataloading_args = dataloading_args

    cfg.predictor_config = parse_args(get_config_cls(cfg.predictor), unknown)

    # For CLI, the `seed` param is passed to the task config and this is not
    # passed to the predictor config, so we have to set it manually
    setattr(cfg.predictor_config, "seed", cfg.seed)

    return cfg, cfg.launcher


if __name__ == "__main__":
    cfg, launcher_opts = cfg_from_cli()
    logger = initialize_logger()

    if cfg.dry_run:
        logger.info(f"Eval config: {cfg}")
        sys.exit(0)

    if launcher_opts:
        from lcm.evaluation.cli import slurm

        slurm.main(cfg, launcher_opts, logger=logger)
    else:
        from lcm.evaluation.cli import local

        local.main(cfg, logger=logger)
