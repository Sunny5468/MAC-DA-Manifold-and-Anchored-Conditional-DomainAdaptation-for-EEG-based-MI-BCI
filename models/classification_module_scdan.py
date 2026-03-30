import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .classification_module import CDANClassificationModule
from .cdan import cdan_multilinear_map, entropy, calc_coeff


class SCDANClassificationModule(CDANClassificationModule):
    """
    SCDAN 训练模块。

    在 CDAN 的基础上增加类条件公共特征对齐损失：
    - 源域使用真实标签计算类中心
    - 目标域使用高置信伪标签计算类中心
    - 对齐同类中心，鼓励学习跨被试公共表示
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
        lambda_common: float = 0.1,
        pseudo_threshold: float = 0.8,
        min_samples_per_class: int = 2,
        use_entropy_conditioning: bool = True,
        use_random_layer: bool = False,
        random_dim: int = 1024,
        lambda_schedule: bool = True,
        lr: float = 0.001,
        lr_discriminator: float = None,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: bool = False,
        max_epochs: int = 1000,
        warmup_epochs: int = 20,
        **kwargs
    ):
        super().__init__(
            model=model,
            n_classes=n_classes,
            d_model=d_model,
            discriminator_hidden_dim=discriminator_hidden_dim,
            discriminator_num_layers=discriminator_num_layers,
            discriminator_dropout=discriminator_dropout,
            lambda_domain=lambda_domain,
            lambda_entropy=lambda_entropy,
            use_entropy_conditioning=use_entropy_conditioning,
            use_random_layer=use_random_layer,
            random_dim=random_dim,
            lambda_schedule=lambda_schedule,
            lr=lr,
            lr_discriminator=lr_discriminator,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            **kwargs
        )
        self.save_hyperparameters(ignore=["model"])

    def _class_conditional_common_loss(self, feat_src, y_src, feat_tgt, logits_tgt):
        probs_tgt = F.softmax(logits_tgt, dim=1)
        conf_tgt, pseudo_tgt = probs_tgt.max(dim=1)
        valid_tgt = conf_tgt >= self.hparams.pseudo_threshold

        per_class_losses = []
        for cls_idx in range(self.hparams.n_classes):
            src_mask = y_src == cls_idx
            tgt_mask = valid_tgt & (pseudo_tgt == cls_idx)

            if src_mask.sum() < self.hparams.min_samples_per_class:
                continue
            if tgt_mask.sum() < self.hparams.min_samples_per_class:
                continue

            src_center = feat_src[src_mask].mean(dim=0)
            tgt_center = feat_tgt[tgt_mask].mean(dim=0)
            per_class_losses.append(F.mse_loss(src_center, tgt_center))

        if not per_class_losses:
            return torch.tensor(0.0, device=feat_src.device)

        return torch.stack(per_class_losses).mean()

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

        domain_pred_src = self.discriminator(self.grl(cond_feat_src))
        domain_pred_tgt = self.discriminator(self.grl(cond_feat_tgt))

        domain_loss = self.cdan_loss(
            domain_pred_src,
            domain_pred_tgt,
            softmax_src,
            softmax_tgt,
        )

        if self.hparams.lambda_entropy > 0:
            entropy_loss = entropy(softmax_tgt, reduction="mean")
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.lambda_common > 0:
            common_loss = self._class_conditional_common_loss(
                feat_src=feat_src,
                y_src=y_src,
                feat_tgt=feat_tgt,
                logits_tgt=logits_tgt,
            )
        else:
            common_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            cls_loss
            + self.hparams.lambda_domain * domain_loss
            + self.hparams.lambda_entropy * entropy_loss
            + self.hparams.lambda_common * common_loss
        )

        acc_src = accuracy(logits_src, y_src, task="multiclass", num_classes=self.hparams.n_classes)

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_loss", domain_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_common_loss", common_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc_src, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lambda", lambda_val, prog_bar=False, on_step=False, on_epoch=True)

        if self.hparams.lambda_entropy > 0:
            self.log("train_entropy_loss", entropy_loss, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss
