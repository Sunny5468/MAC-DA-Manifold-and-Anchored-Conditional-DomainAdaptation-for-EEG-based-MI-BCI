import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassConfusionMatrix

import pytorch_lightning as pl

from utils.lr_scheduler import linear_warmup_cosine_decay


class DeepCORALClassificationModule(pl.LightningModule):
    """
    Deep CORAL baseline (Sun & Saenko, 2016).

    Loss = CrossEntropy(source) + lambda_coral * CORAL(feat_src, feat_tgt)

    CORAL loss aligns the second-order statistics (covariance) of source and
    target feature distributions *without* any adversarial training.
    No GRL or domain discriminator is used.
    """

    def __init__(
        self,
        model,
        n_classes: int,
        d_model: int = 32,
        lambda_coral: float = 1.0,
        lr: float = 0.001,
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

        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_confmat = None

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def get_features_and_logits(self, x):
        return self.model.get_features_and_logits(x)

    # ------------------------------------------------------------------
    # CORAL loss
    # ------------------------------------------------------------------
    @staticmethod
    def _coral_loss(feat_src: torch.Tensor, feat_tgt: torch.Tensor) -> torch.Tensor:
        """
        Original Deep CORAL loss.
        L_CORAL = 1/(4*d^2) * || C_S - C_T ||_F^2
        """
        d = feat_src.size(1)

        def _cov(x):
            n = x.size(0)
            if n <= 1:
                return torch.zeros(d, d, device=x.device, dtype=x.dtype)
            x_c = x - x.mean(dim=0, keepdim=True)
            return (x_c.T @ x_c) / (n - 1)

        cov_src = _cov(feat_src)
        cov_tgt = _cov(feat_tgt)
        loss = torch.sum((cov_src - cov_tgt) ** 2) / (4 * d * d)
        return loss

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        params = list(self.model.parameters())

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr,
                                         betas=betas, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr,
                                          betas=betas, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} not implemented")

        if self.hparams.scheduler:
            scheduler = LambdaLR(
                optimizer,
                linear_warmup_cosine_decay(self.hparams.warmup_epochs, self.hparams.max_epochs),
            )
            return [optimizer], [scheduler]
        return [optimizer]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        (x_src, y_src), (x_tgt, _) = batch

        feat_src, logits_src = self.get_features_and_logits(x_src)
        feat_tgt, _logits_tgt = self.get_features_and_logits(x_tgt)

        cls_loss = F.cross_entropy(logits_src, y_src)
        coral_loss = self._coral_loss(feat_src, feat_tgt)
        total_loss = cls_loss + self.hparams.lambda_coral * coral_loss

        acc_src = accuracy(logits_src, y_src, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("train_loss",       total_loss,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_cls_loss",   cls_loss,    prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_coral_loss", coral_loss,  prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc",        acc_src,     prog_bar=True,  on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("val_loss", loss, prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_acc",  acc,  prog_bar=True,  on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("test_loss",  loss, prog_bar=True,  on_step=False, on_epoch=True)
        self.log("test_acc",   acc,  prog_bar=True,  on_step=False, on_epoch=True)

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
