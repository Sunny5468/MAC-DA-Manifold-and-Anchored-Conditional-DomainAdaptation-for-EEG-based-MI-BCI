import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .classification_module import CDANClassificationModule
from .cdan import cdan_multilinear_map, entropy, calc_coeff


class CDANCCCORALClassificationModule(CDANClassificationModule):
    """
    Baseline CDAN + Class-Conditional CORAL (CC-CORAL).

    只在 baseline CDAN 上增加一个可控的类条件二阶统计对齐项，
    不改动原有 CDAN 模块文件。
    """

    def __init__(
        self,
        *args,
        lambda_cccoral: float = 0.01,
        cccoral_alpha: float = 0.1,
        pseudo_threshold: float = 0.9,
        min_samples_per_class: int = 4,
        cccoral_warmup_epochs: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_cccoral = float(lambda_cccoral)
        self.cccoral_alpha = float(cccoral_alpha)
        self.pseudo_threshold = float(pseudo_threshold)
        self.min_samples_per_class = int(min_samples_per_class)
        self.cccoral_warmup_epochs = int(cccoral_warmup_epochs)

    @staticmethod
    def _covariance(x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return torch.zeros(x.size(1), x.size(1), device=x.device, dtype=x.dtype)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return (x_centered.T @ x_centered) / (x.size(0) - 1)

    def _cccoral_loss(
        self,
        feat_src: torch.Tensor,
        y_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        softmax_tgt: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = int(self.hparams.n_classes)
        conf_tgt, pseudo_tgt = torch.max(softmax_tgt, dim=1)

        total_loss = torch.tensor(0.0, device=feat_src.device)
        valid_classes = 0

        for class_idx in range(n_classes):
            src_mask = y_src == class_idx
            tgt_mask = (pseudo_tgt == class_idx) & (conf_tgt >= self.pseudo_threshold)

            src_count = int(src_mask.sum().item())
            tgt_count = int(tgt_mask.sum().item())

            if src_count < self.min_samples_per_class or tgt_count < self.min_samples_per_class:
                continue

            src_feat_c = feat_src[src_mask]
            tgt_feat_c = feat_tgt[tgt_mask]

            src_mean = src_feat_c.mean(dim=0)
            tgt_mean = tgt_feat_c.mean(dim=0)
            mean_loss = torch.mean((src_mean - tgt_mean) ** 2)

            src_cov = self._covariance(src_feat_c)
            tgt_cov = self._covariance(tgt_feat_c)
            cov_loss = torch.mean((src_cov - tgt_cov) ** 2)

            total_loss = total_loss + cov_loss + self.cccoral_alpha * mean_loss
            valid_classes += 1

        if valid_classes == 0:
            return torch.tensor(0.0, device=feat_src.device)
        return total_loss / valid_classes

    def training_step(self, batch, batch_idx):
        (x_src, y_src), (x_tgt, _) = batch

        if self.hparams.lambda_schedule:
            lambda_val = calc_coeff(self.current_iteration, max_iter=self.max_iterations)
            self.grl.set_lambda(lambda_val)
        else:
            lambda_val = self.hparams.lambda_domain

        self.current_iteration += 1

        feat_src, logits_src = self.get_features_and_logits(x_src)
        softmax_src = F.softmax(logits_src, dim=1)

        feat_tgt, logits_tgt = self.get_features_and_logits(x_tgt)
        softmax_tgt = F.softmax(logits_tgt, dim=1)

        cls_loss = F.cross_entropy(logits_src, y_src)

        if self.random_layer is not None:
            cond_feat_src = self.random_layer(feat_src, softmax_src)
            cond_feat_tgt = self.random_layer(feat_tgt, softmax_tgt)
        else:
            cond_feat_src = cdan_multilinear_map(feat_src, softmax_src)
            cond_feat_tgt = cdan_multilinear_map(feat_tgt, softmax_tgt)

        cond_feat_src_grl = self.grl(cond_feat_src)
        cond_feat_tgt_grl = self.grl(cond_feat_tgt)

        domain_pred_src = self.discriminator(cond_feat_src_grl)
        domain_pred_tgt = self.discriminator(cond_feat_tgt_grl)

        domain_loss = self.cdan_loss(domain_pred_src, domain_pred_tgt, softmax_src, softmax_tgt)

        if self.hparams.lambda_entropy > 0:
            entropy_loss = entropy(softmax_tgt, reduction="mean")
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        if self.current_epoch >= self.cccoral_warmup_epochs and self.lambda_cccoral > 0:
            cccoral_loss = self._cccoral_loss(feat_src, y_src, feat_tgt, softmax_tgt)
        else:
            cccoral_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            cls_loss
            + self.hparams.lambda_domain * domain_loss
            + self.hparams.lambda_entropy * entropy_loss
            + self.lambda_cccoral * cccoral_loss
        )

        acc_src = accuracy(logits_src, y_src, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_loss", domain_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc_src, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lambda", lambda_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_cccoral_loss", cccoral_loss, prog_bar=False, on_step=False, on_epoch=True)

        if self.hparams.lambda_entropy > 0:
            self.log("train_entropy_loss", entropy_loss, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss
