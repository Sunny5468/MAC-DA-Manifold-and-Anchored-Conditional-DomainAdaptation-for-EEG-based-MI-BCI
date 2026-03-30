import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassConfusionMatrix

import pytorch_lightning as pl

from utils.lr_scheduler import linear_warmup_cosine_decay
from .cdan import (
    GradientReversalLayer,
    DomainDiscriminator,
    RandomLayer,
    cdan_multilinear_map,
    entropy,
    calc_coeff,
)


def _coral_loss(feat_src: torch.Tensor, feat_tgt: torch.Tensor) -> torch.Tensor:
    """
    CORAL moment alignment on feature covariance.
    A light-weight and commonly used batch-level alignment term.
    """
    if feat_src.size(0) < 2 or feat_tgt.size(0) < 2:
        return torch.tensor(0.0, device=feat_src.device, dtype=feat_src.dtype)

    src_centered = feat_src - feat_src.mean(dim=0, keepdim=True)
    tgt_centered = feat_tgt - feat_tgt.mean(dim=0, keepdim=True)

    cov_src = (src_centered.t() @ src_centered) / (feat_src.size(0) - 1)
    cov_tgt = (tgt_centered.t() @ tgt_centered) / (feat_tgt.size(0) - 1)

    d = feat_src.size(1)
    return torch.sum((cov_src - cov_tgt) ** 2) / (4.0 * d * d)


class MIDABANClassificationModule(pl.LightningModule):
    """
    MI-DABAN (typical variant for MI-EEG UDA):
    - Source supervised CE
    - Global feature adversarial alignment (DANN branch)
    - Conditional adversarial alignment (CDAN-style branch)
    - Batch moment alignment (CORAL)
    - Optional target entropy minimization
    """

    def __init__(
        self,
        model,
        n_classes: int,
        d_model: int = 32,
        discriminator_hidden_dim: int = 256,
        discriminator_num_layers: int = 2,
        discriminator_dropout: float = 0.5,
        lambda_domain: float = 1.0,
        lambda_conditional: float = 0.5,
        lambda_moment: float = 0.1,
        lambda_entropy: float = 0.0,
        lambda_schedule: bool = True,
        use_random_layer: bool = False,
        random_dim: int = 1024,
        lr: float = 0.001,
        lr_discriminator: float = None,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: bool = False,
        max_epochs: int = 1000,
        warmup_epochs: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.discriminator_global = DomainDiscriminator(
            input_dim=d_model,
            hidden_dim=discriminator_hidden_dim,
            num_layers=discriminator_num_layers,
            dropout=discriminator_dropout,
        )

        cond_dim = random_dim if use_random_layer else d_model * n_classes
        self.discriminator_conditional = DomainDiscriminator(
            input_dim=cond_dim,
            hidden_dim=discriminator_hidden_dim,
            num_layers=discriminator_num_layers,
            dropout=discriminator_dropout,
        )

        self.random_layer = None
        if use_random_layer:
            self.random_layer = RandomLayer([d_model, n_classes], random_dim)

        self.grl_global = GradientReversalLayer(lambda_=lambda_domain)
        self.grl_conditional = GradientReversalLayer(lambda_=lambda_conditional)

        self.current_iteration = 0
        self.max_iterations = max_epochs * 100

        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_confmat = None

    def forward(self, x):
        return self.model(x)

    def get_features_and_logits(self, x):
        return self.model.get_features_and_logits(x)

    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)

        model_params = list(self.model.parameters())
        discr_global_params = list(self.discriminator_global.parameters())
        discr_cond_params = list(self.discriminator_conditional.parameters())

        lr = self.hparams.lr
        lr_d = self.hparams.lr_discriminator if self.hparams.lr_discriminator else lr

        param_groups = [
            {"params": model_params, "lr": lr},
            {"params": discr_global_params, "lr": lr_d},
            {"params": discr_cond_params, "lr": lr_d},
        ]

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} not implemented")

        if self.hparams.scheduler:
            scheduler = LambdaLR(
                optimizer,
                linear_warmup_cosine_decay(self.hparams.warmup_epochs, self.hparams.max_epochs),
            )
            return [optimizer], [scheduler]
        return [optimizer]

    @staticmethod
    def _domain_loss(domain_pred_src: torch.Tensor, domain_pred_tgt: torch.Tensor) -> torch.Tensor:
        device = domain_pred_src.device
        domain_label_src = torch.zeros(domain_pred_src.size(0), 1, device=device)
        domain_label_tgt = torch.ones(domain_pred_tgt.size(0), 1, device=device)

        loss_src = F.binary_cross_entropy_with_logits(domain_pred_src, domain_label_src)
        loss_tgt = F.binary_cross_entropy_with_logits(domain_pred_tgt, domain_label_tgt)
        return loss_src + loss_tgt

    def _conditional_features(self, features: torch.Tensor, softmax_output: torch.Tensor) -> torch.Tensor:
        if self.random_layer is not None:
            return self.random_layer(features, softmax_output)
        return cdan_multilinear_map(features, softmax_output)

    def training_step(self, batch, batch_idx):
        (x_src, y_src), (x_tgt, _) = batch

        if self.hparams.lambda_schedule:
            coeff = calc_coeff(self.current_iteration, max_iter=self.max_iterations)
        else:
            coeff = 1.0

        lambda_global = self.hparams.lambda_domain * coeff
        lambda_cond = self.hparams.lambda_conditional * coeff

        self.grl_global.set_lambda(lambda_global)
        self.grl_conditional.set_lambda(lambda_cond)
        self.current_iteration += 1

        feat_src, logits_src = self.get_features_and_logits(x_src)
        feat_tgt, logits_tgt = self.get_features_and_logits(x_tgt)

        softmax_src = F.softmax(logits_src, dim=1)
        softmax_tgt = F.softmax(logits_tgt, dim=1)

        cls_loss = F.cross_entropy(logits_src, y_src)

        pred_dom_src_global = self.discriminator_global(self.grl_global(feat_src))
        pred_dom_tgt_global = self.discriminator_global(self.grl_global(feat_tgt))
        loss_domain_global = self._domain_loss(pred_dom_src_global, pred_dom_tgt_global)

        cond_src = self._conditional_features(feat_src, softmax_src.detach())
        cond_tgt = self._conditional_features(feat_tgt, softmax_tgt.detach())
        pred_dom_src_cond = self.discriminator_conditional(self.grl_conditional(cond_src))
        pred_dom_tgt_cond = self.discriminator_conditional(self.grl_conditional(cond_tgt))
        loss_domain_cond = self._domain_loss(pred_dom_src_cond, pred_dom_tgt_cond)

        loss_moment = _coral_loss(feat_src, feat_tgt)

        if self.hparams.lambda_entropy > 0:
            loss_entropy = entropy(softmax_tgt, reduction="mean")
        else:
            loss_entropy = torch.tensor(0.0, device=self.device)

        total_loss = (
            cls_loss
            + lambda_global * loss_domain_global
            + lambda_cond * loss_domain_cond
            + self.hparams.lambda_moment * loss_moment
            + self.hparams.lambda_entropy * loss_entropy
        )

        acc_src = accuracy(logits_src, y_src, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_global", loss_domain_global, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_cond", loss_domain_cond, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_moment_loss", loss_moment, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc_src, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lambda_adv_coeff", coeff, prog_bar=False, on_step=False, on_epoch=True)

        if self.hparams.lambda_entropy > 0:
            self.log("train_entropy_loss", loss_entropy, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        preds = torch.argmax(y_hat, dim=-1)
        self.test_kappa.update(preds, y)
        self.test_cm.update(preds, y)
        self.log("test_kappa", self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)

        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self):
        cm_counts = self.test_cm.compute()
        self.test_cm.reset()

        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0

        self.test_confmat = cm_percent.cpu()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)

    def on_train_epoch_start(self):
        if hasattr(self.trainer, "num_training_batches"):
            self.max_iterations = self.hparams.max_epochs * self.trainer.num_training_batches
