"""
EA-enabled training entrypoint (non-intrusive wrapper).

This file does not modify the original train_pipeline.py.
It reuses all original CLI/training logic, swaps in EA-capable
datamodule class resolution, and appends "_ea" to dataset_name
when EA preprocessing is enabled so output directories are explicit.
"""

import copy
import argparse
import sys

import train_pipeline as _base
from pytorch_lightning import Trainer as _PLTrainer
from utils.joint_training_gate import JointCollapseEarlyStopGate
from utils.get_datamodule_cls_ea import get_datamodule_cls as _get_datamodule_cls_ea


# Module-level flag set from CLI; read by _with_ea_suffix.
_ea_gate_enabled = False


def _with_ea_suffix(config: dict) -> dict:
    """Return a config copy whose dataset_name is suffixed with '_ea' when EA is on."""
    cfg = copy.deepcopy(config)
    use_ea = bool(cfg.get("preprocessing", {}).get("ea", False))
    if use_ea:
        dataset_name = str(cfg.get("dataset_name", ""))
        if dataset_name and not dataset_name.endswith("_ea"):
            cfg["dataset_name"] = f"{dataset_name}_ea"
    # Inject ea_gate flag so the datamodule can read it.
    if _ea_gate_enabled and isinstance(cfg.get("preprocessing"), dict):
        cfg["preprocessing"]["ea_gate"] = True
    return cfg


# Inject EA-capable datamodule resolver before running.
_base.get_datamodule_cls = _get_datamodule_cls_ea


# Wrap training entrypoints so the result directory name includes EA state.
_orig_train_standard = _base.train_and_test_standard
_orig_train_cdan = _base.train_and_test_cdan
_orig_train_dann = _base.train_and_test_dann


def _train_and_test_standard_ea_named(config):
    return _orig_train_standard(_with_ea_suffix(config))


def _train_and_test_cdan_ea_named(
    config,
    use_v2=False,
    use_v2_simple=False,
    use_cccoral=False,
    use_scdan=False,
):
    return _orig_train_cdan(
        _with_ea_suffix(config),
        use_v2=use_v2,
        use_v2_simple=use_v2_simple,
        use_cccoral=use_cccoral,
        use_scdan=use_scdan,
    )


def _train_and_test_dann_ea_named(config):
    return _orig_train_dann(_with_ea_suffix(config))


_base.train_and_test_standard = _train_and_test_standard_ea_named
_base.train_and_test_cdan = _train_and_test_cdan_ea_named
_base.train_and_test_dann = _train_and_test_dann_ea_named


class _TrainerWithJointGate(_PLTrainer):
    """EA wrapper trainer that appends a joint early-stop/collapse gate callback."""

    def __init__(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks", []))
        callbacks.append(
            JointCollapseEarlyStopGate(
                warmup_no_stop=20,
                monitor="val_acc",
                min_delta=0.002,
                early_stop_patience=12,
                collapse_patience=5,
                entropy_norm_th=0.20,
                class_max_ratio_th=0.80,
                probe_target_batches=2,
            )
        )
        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)


class _TrainerWithProgressBar(_PLTrainer):
    """EA wrapper trainer that forces progress bar on."""

    def __init__(self, *args, **kwargs):
        kwargs["enable_progress_bar"] = True
        super().__init__(*args, **kwargs)


class _TrainerWithJointGateAndProgressBar(_PLTrainer):
    """EA wrapper trainer that enables both joint gate and progress bar."""

    def __init__(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks", []))
        callbacks.append(
            JointCollapseEarlyStopGate(
                warmup_no_stop=20,
                monitor="val_acc",
                min_delta=0.002,
                early_stop_patience=12,
                collapse_patience=5,
                entropy_norm_th=0.20,
                class_max_ratio_th=0.80,
                probe_target_batches=2,
            )
        )
        kwargs["callbacks"] = callbacks
        kwargs["enable_progress_bar"] = True
        super().__init__(*args, **kwargs)


def _parse_wrapper_args(argv):
    """Parse EA wrapper-only args and keep remaining args for base pipeline."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--joint_gate",
        dest="joint_gate",
        action="store_true",
        help="Enable joint collapse+early-stop gate callback",
    )
    parser.add_argument(
        "--no_joint_gate",
        dest="joint_gate",
        action="store_false",
        help="Disable joint collapse+early-stop gate callback",
    )
    parser.add_argument(
        "--progress_bar",
        dest="progress_bar",
        action="store_true",
        help="Force-enable training progress bar",
    )
    parser.add_argument(
        "--no_progress_bar",
        dest="progress_bar",
        action="store_false",
        help="Disable wrapper-forced progress bar",
    )
    parser.add_argument(
        "--gate",
        dest="ea_gate",
        action="store_true",
        help="Enable adaptive EA hard gating (1-sigma log-det rule). "
             "When target subject's covariance log-determinant falls below "
             "mu-sigma, EA is disabled for the entire fold.",
    )
    parser.add_argument(
        "--no_gate",
        dest="ea_gate",
        action="store_false",
        help="Disable adaptive EA hard gating",
    )
    parser.set_defaults(joint_gate=False)
    parser.set_defaults(progress_bar=False)
    parser.set_defaults(ea_gate=False)
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    wrapper_args, remaining = _parse_wrapper_args(sys.argv[1:])

    # Forward only base-supported args to the original parser.
    sys.argv = [sys.argv[0], *remaining]

    # Inject ea_gate flag into preprocessing dict via the _with_ea_suffix wrapper.
    import train_pipeline_ea as _self
    _self._ea_gate_enabled = wrapper_args.ea_gate
    if wrapper_args.ea_gate:
        print("[EA Wrapper] EA hard gating enabled (1-sigma log-det rule)")
    _base_run_fn = _base.run

    # Keep the switch but disable by default as requested.
    if wrapper_args.joint_gate and wrapper_args.progress_bar:
        _base.Trainer = _TrainerWithJointGateAndProgressBar
        print("[EA Wrapper] joint gate enabled; progress bar enabled")
    elif wrapper_args.joint_gate:
        _base.Trainer = _TrainerWithJointGate
        print("[EA Wrapper] joint gate enabled")
    elif wrapper_args.progress_bar:
        _base.Trainer = _TrainerWithProgressBar
        print("[EA Wrapper] joint gate disabled (default); progress bar enabled")
    else:
        print("[EA Wrapper] joint gate disabled (default)")

    _base_run_fn()
