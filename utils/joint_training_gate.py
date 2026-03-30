from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


@dataclass
class GateState:
    best_score: float = float("-inf")
    bad_epochs: int = 0
    collapse_epochs: int = 0
    stop_reason: str = ""


class JointCollapseEarlyStopGate(Callback):
    """
    Joint gate for conservative early stop + collapse detection.

     Stop conditions after warmup:
     1) val metric plateau for `early_stop_patience` epochs.
     2) collapse pattern persists for `collapse_patience` epochs:
         low normalized entropy + high single-class dominance on
         unlabeled target-train batches.

     This uses only train-time available signals and never test-set feedback,
     suitable for strict UDA evaluation.
    """

    def __init__(
        self,
        warmup_no_stop: int = 20,
        monitor: str = "val_acc",
        min_delta: float = 0.002,
        early_stop_patience: int = 12,
        collapse_patience: int = 5,
        entropy_norm_th: float = 0.20,
        class_max_ratio_th: float = 0.80,
        probe_target_batches: int = 2,
        trace_log_dir: str = "logs",
    ):
        super().__init__()
        self.warmup_no_stop = int(warmup_no_stop)
        self.monitor = monitor
        self.min_delta = float(min_delta)
        self.early_stop_patience = int(early_stop_patience)
        self.collapse_patience = int(collapse_patience)
        self.entropy_norm_th = float(entropy_norm_th)
        self.class_max_ratio_th = float(class_max_ratio_th)
        self.probe_target_batches = int(probe_target_batches)
        self.trace_log_dir = trace_log_dir
        self.trace_file_path: Optional[Path] = None
        self.state = GateState()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.state = GateState()
        self.trace_file_path = None
        # Create the trace file immediately so users can observe logging from epoch 0.
        self._ensure_trace_file(trainer)

    def _ensure_trace_file(self, trainer: pl.Trainer) -> Path:
        if self.trace_file_path is not None:
            return self.trace_file_path

        subject_id = getattr(getattr(trainer, "datamodule", None), "subject_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.trace_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file_path = log_dir / f"joint_gate_trace_subject{subject_id}_{timestamp}.csv"

        with self.trace_file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "monitor",
                "monitor_value",
                "best_score",
                "bad_epochs",
                "entropy_norm",
                "class_max_ratio",
                "collapse_hit",
                "collapse_epochs",
                "plateau_stop",
                "collapse_stop",
                "stop_reason",
            ])

        return self.trace_file_path

    def _append_trace_row(
        self,
        trainer: pl.Trainer,
        epoch: int,
        monitor_value: Optional[float],
        entropy_norm: Optional[float],
        class_max_ratio: Optional[float],
        collapse_hit: bool,
        plateau_stop: bool,
        collapse_stop: bool,
    ) -> None:
        trace_path = self._ensure_trace_file(trainer)
        with trace_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                self.monitor,
                "" if monitor_value is None else f"{monitor_value:.6f}",
                f"{self.state.best_score:.6f}",
                self.state.bad_epochs,
                "" if entropy_norm is None else f"{entropy_norm:.6f}",
                "" if class_max_ratio is None else f"{class_max_ratio:.6f}",
                int(collapse_hit),
                self.state.collapse_epochs,
                int(plateau_stop),
                int(collapse_stop),
                self.state.stop_reason,
            ])

    def _compute_target_stats(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Optional[tuple[float, float]]:
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "target_train_dataloader"):
            return None

        loader = dm.target_train_dataloader()
        if loader is None:
            return None

        pl_module_was_training = pl_module.training
        pl_module.eval()

        entropies = []
        pred_counts = None
        total = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if self.probe_target_batches > 0 and i >= self.probe_target_batches:
                    break

                x, _ = batch
                x = x.to(pl_module.device)
                logits = pl_module(x)
                probs = torch.softmax(logits, dim=1)

                n_classes = probs.shape[1]
                log_base = torch.log(torch.tensor(float(n_classes), device=probs.device))
                ent = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1) / log_base
                entropies.append(ent.mean().item())

                preds = torch.argmax(logits, dim=1)
                counts = torch.bincount(preds, minlength=n_classes).float()
                pred_counts = counts if pred_counts is None else pred_counts + counts
                total += int(preds.numel())

        if pl_module_was_training:
            pl_module.train()

        if not entropies or pred_counts is None or total <= 0:
            return None

        entropy_norm = float(sum(entropies) / len(entropies))
        class_max_ratio = float((pred_counts / pred_counts.sum().clamp_min(1.0)).max().item())
        return entropy_norm, class_max_ratio

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = int(trainer.current_epoch)
        metrics = trainer.callback_metrics

        monitored = metrics.get(self.monitor)
        monitor_value: Optional[float] = None
        if monitored is not None:
            monitor_value = float(monitored.detach().cpu().item())

        if epoch < self.warmup_no_stop:
            # Still record warmup epochs for full trace visibility.
            self._append_trace_row(
                trainer=trainer,
                epoch=epoch,
                monitor_value=monitor_value,
                entropy_norm=None,
                class_max_ratio=None,
                collapse_hit=False,
                plateau_stop=False,
                collapse_stop=False,
            )
            return

        if monitored is not None:
            score = monitor_value
            assert score is not None
            if score > (self.state.best_score + self.min_delta):
                self.state.best_score = score
                self.state.bad_epochs = 0
            else:
                self.state.bad_epochs += 1

        collapse_hit = False
        entropy_norm: Optional[float] = None
        class_max_ratio: Optional[float] = None
        target_stats = self._compute_target_stats(trainer, pl_module)
        if target_stats is not None:
            entropy_norm, class_max_ratio = target_stats
            collapse_hit = (
                entropy_norm < self.entropy_norm_th
                and class_max_ratio > self.class_max_ratio_th
            )

            pl_module.log("gate_entropy_norm", entropy_norm, prog_bar=False, on_step=False, on_epoch=True)
            pl_module.log("gate_class_max_ratio", class_max_ratio, prog_bar=False, on_step=False, on_epoch=True)

        if collapse_hit:
            self.state.collapse_epochs += 1
        else:
            self.state.collapse_epochs = 0

        plateau_stop = self.state.bad_epochs >= self.early_stop_patience
        collapse_stop = self.state.collapse_epochs >= self.collapse_patience

        self._append_trace_row(
            trainer=trainer,
            epoch=epoch,
            monitor_value=monitor_value,
            entropy_norm=entropy_norm,
            class_max_ratio=class_max_ratio,
            collapse_hit=collapse_hit,
            plateau_stop=plateau_stop,
            collapse_stop=collapse_stop,
        )

        if plateau_stop or collapse_stop:
            if collapse_stop:
                self.state.stop_reason = "collapse_gate"
            else:
                self.state.stop_reason = "early_stop_plateau"

            print(
                "[JointGate] Stop triggered: "
                f"reason={self.state.stop_reason}, epoch={epoch}, "
                f"bad_epochs={self.state.bad_epochs}, "
                f"collapse_epochs={self.state.collapse_epochs}, "
                f"entropy_norm_th={self.entropy_norm_th}, "
                f"class_max_ratio_th={self.class_max_ratio_th}"
            )
            trainer.should_stop = True
