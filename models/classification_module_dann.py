import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassConfusionMatrix

import pytorch_lightning as pl

from utils.lr_scheduler import linear_warmup_cosine_decay
from .cdan import GradientReversalLayer, DomainDiscriminator, entropy, calc_coeff


class DANNClassificationModule(pl.LightningModule):
    """
    DANN 训练模块（baseline）

    与 CDAN 区别：
    - DANN 仅使用特征进行域对抗，不使用 feature-logit 条件外积。
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
        lambda_entropy: float = 0.0,
        lambda_schedule: bool = True,
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

        self.discriminator = DomainDiscriminator(
            input_dim=d_model,
            hidden_dim=discriminator_hidden_dim,
            num_layers=discriminator_num_layers,
            dropout=discriminator_dropout,
        )

        self.grl = GradientReversalLayer(lambda_=lambda_domain)

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
        discriminator_params = list(self.discriminator.parameters())

        lr = self.hparams.lr
        lr_d = self.hparams.lr_discriminator if self.hparams.lr_discriminator else lr

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                [
                    {"params": model_params, "lr": lr},
                    {"params": discriminator_params, "lr": lr_d},
                ],
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(
                [
                    {"params": model_params, "lr": lr},
                    {"params": discriminator_params, "lr": lr_d},
                ],
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {"params": model_params, "lr": lr},
                    {"params": discriminator_params, "lr": lr_d},
                ],
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

    def _domain_loss(self, domain_pred_src: torch.Tensor, domain_pred_tgt: torch.Tensor) -> torch.Tensor:
        device = domain_pred_src.device
        domain_label_src = torch.zeros(domain_pred_src.size(0), 1, device=device)
        domain_label_tgt = torch.ones(domain_pred_tgt.size(0), 1, device=device)

        loss_src = F.binary_cross_entropy_with_logits(domain_pred_src, domain_label_src)
        loss_tgt = F.binary_cross_entropy_with_logits(domain_pred_tgt, domain_label_tgt)
        return loss_src + loss_tgt

    def training_step(self, batch, batch_idx):
        (x_src, y_src), (x_tgt, _) = batch

        if self.hparams.lambda_schedule:
            lambda_val = calc_coeff(self.current_iteration, max_iter=self.max_iterations)
            self.grl.set_lambda(lambda_val)
        else:
            lambda_val = self.hparams.lambda_domain

        self.current_iteration += 1

        feat_src, logits_src = self.get_features_and_logits(x_src)
        feat_tgt, logits_tgt = self.get_features_and_logits(x_tgt)

        softmax_tgt = F.softmax(logits_tgt, dim=1)

        cls_loss = F.cross_entropy(logits_src, y_src)

        domain_pred_src = self.discriminator(self.grl(feat_src))
        domain_pred_tgt = self.discriminator(self.grl(feat_tgt))
        domain_loss = self._domain_loss(domain_pred_src, domain_pred_tgt)

        if self.hparams.lambda_entropy > 0:
            entropy_loss = entropy(softmax_tgt, reduction="mean")
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            cls_loss
            + self.hparams.lambda_domain * domain_loss
            + self.hparams.lambda_entropy * entropy_loss
        )

        acc_src = accuracy(logits_src, y_src, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_loss", domain_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc_src, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lambda", lambda_val, prog_bar=False, on_step=False, on_epoch=True)

        if self.hparams.lambda_entropy > 0:
            self.log("train_entropy_loss", entropy_loss, prog_bar=False, on_step=False, on_epoch=True)

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
