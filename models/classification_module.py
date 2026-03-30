import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import (
    MulticlassCohenKappa, MulticlassConfusionMatrix
)

import pytorch_lightning as pl
from utils.lr_scheduler import linear_warmup_cosine_decay
import random

from .cdan import (
    GradientReversalLayer, DomainDiscriminator, 
    cdan_multilinear_map, entropy, calc_coeff, CDANLoss
)


# Helper: Randomly selects a subset of EEG channels (augmentations)
def select_random_channels(x, keep_ratio=0.9):
    """
    Select a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: fraction of channels to keep
    Returns:
        Tensor of shape [B, C_selected, T]
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    return x[:, keep_indices, :], keep_indices


# Helper: Randomly masks EEG channels (augmentations)
def random_channel_mask(x, keep_ratio=0.9):
    """
    Randomly keeps a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: Float, ratio of channels to keep (e.g., 0.9 to keep 90%).
    Returns:
        Augmented tensor with masked channels set to 0.
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    mask = torch.zeros_like(x)
    mask[:, keep_indices, :] = 1
    return x * mask


# Lightning module for standard classification
class ClassificationModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_classes,
            lr=0.001,
            weight_decay=0.0,
            optimizer="adam",
            scheduler=False,
            max_epochs=1000,
            warmup_epochs=20,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        # ── metrics ───────────────────────────────────────
        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)        
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)  
        # will hold the final cm on CPU after test
        self.test_confmat = None

    # forward
    def forward(self, x):
        return self.model(x)

    # optimiser / scheduler
    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # steps
    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        return {"test_loss": loss, "test_acc": acc}

    # common logic
    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        if mode == "train":
            if self.hparams.get("random_channel_masking", False):
                # Add random EEG channel masking augmentation (did not improve the training)
                x = random_channel_mask(x, self.hparams.get("keep_ratio",0.9))
            if self.hparams.get("random_channel_selection", False):
                # Randomly select a subset of EEG channels (did not improve the training)
                x, _ = select_random_channels(x, self.hparams.get("keep_ratio",0.9))

        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        # log scalar metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        if mode == "test":
            preds = torch.argmax(y_hat, dim=-1)
            # ── update epoch-level metrics ───────────────────────────
            self.test_kappa.update(preds, y)                       # accumulate
            self.test_cm.update(preds, y)
            
            self.log("test_kappa", self.test_kappa,                # Lightning will call .compute()
                    prog_bar=False, on_step=False, on_epoch=True)

        return loss, acc
    
    # grab confusion matrix once per test epoch
    def on_test_epoch_end(self):
        # 1) raw counts  ───────────────────────────────────────────
        cm_counts = self.test_cm.compute()   # shape [C, C]
        self.test_cm.reset()

        # 2) row-normalise → %  (handle rows with 0 samples safely)
        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0

        self.test_confmat = cm_percent.cpu()        # stash for plotting

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)


# Lightning module for CDAN (Conditional Domain Adversarial Network)
class CDANClassificationModule(pl.LightningModule):
    """
    CDAN 训练模块
    
    实现条件域对抗网络训练，用于跨被试迁移学习
    
    Args:
        model: ATCNetModule 实例（包含 feature_extractor 和 classifier）
        n_classes: 类别数
        d_model: 特征维度
        discriminator_hidden_dim: 域判别器隐藏层维度
        discriminator_num_layers: 域判别器层数
        discriminator_dropout: 域判别器 dropout
        lambda_domain: 域对抗损失权重
        lambda_entropy: 熵最小化损失权重
        use_entropy_conditioning: 是否使用熵条件对抗 (CDAN+E)
        use_random_layer: 是否使用随机层降维
        random_dim: 随机层输出维度
        lambda_schedule: 是否使用 lambda 调度
        lr: 学习率
        lr_discriminator: 域判别器学习率（如果为 None，则使用 lr）
        weight_decay: 权重衰减
        optimizer: 优化器类型
        scheduler: 是否使用学习率调度
        max_epochs: 最大训练轮数
        warmup_epochs: 预热轮数
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
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        # 主模型（特征提取器 + 分类器）
        self.model = model
        
        # 计算域判别器输入维度
        # CDAN 使用特征和分类预测的外积，维度为 d_model * n_classes
        if use_random_layer:
            discriminator_input_dim = random_dim
        else:
            discriminator_input_dim = d_model * n_classes
        
        # 域判别器
        self.discriminator = DomainDiscriminator(
            input_dim=discriminator_input_dim,
            hidden_dim=discriminator_hidden_dim,
            num_layers=discriminator_num_layers,
            dropout=discriminator_dropout
        )
        
        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_=lambda_domain)
        
        # CDAN 损失
        self.cdan_loss = CDANLoss(
            use_entropy=use_entropy_conditioning,
            use_random_layer=use_random_layer,
            d_model=d_model,
            n_classes=n_classes,
            random_dim=random_dim
        )
        
        # 随机层（可选）
        if use_random_layer:
            from .cdan import RandomLayer
            self.random_layer = RandomLayer([d_model, n_classes], random_dim)
        else:
            self.random_layer = None
        
        # 训练状态
        self.current_iteration = 0
        self.max_iterations = max_epochs * 100  # 估计值，会在训练时更新
        
        # ── metrics ───────────────────────────────────────
        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)        
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)  
        self.test_confmat = None

    def forward(self, x):
        """标准前向传播，返回分类结果"""
        return self.model(x)
    
    def get_features_and_logits(self, x):
        """获取特征和分类 logits"""
        return self.model.get_features_and_logits(x)

    def configure_optimizers(self):
        """配置优化器"""
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        
        # 主模型参数
        model_params = list(self.model.parameters())
        
        # 域判别器参数
        discriminator_params = list(self.discriminator.parameters())
        
        # 学习率
        lr = self.hparams.lr
        lr_d = self.hparams.lr_discriminator if self.hparams.lr_discriminator else lr
        
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam([
                {'params': model_params, 'lr': lr},
                {'params': discriminator_params, 'lr': lr_d}
            ], betas=betas, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW([
                {'params': model_params, 'lr': lr},
                {'params': discriminator_params, 'lr': lr_d}
            ], betas=betas, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD([
                {'params': model_params, 'lr': lr},
                {'params': discriminator_params, 'lr': lr_d}
            ], weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} not implemented")
        
        if self.hparams.scheduler:
            scheduler = LambdaLR(
                optimizer,
                linear_warmup_cosine_decay(self.hparams.warmup_epochs, self.hparams.max_epochs)
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        """
        CDAN 训练步骤
        
        batch 格式: ((x_src, y_src), (x_tgt, y_tgt))
        其中 y_tgt 在训练时不使用（无监督域适应）
        """
        # 解析 batch
        (x_src, y_src), (x_tgt, _) = batch
        
        # 更新 lambda（如果使用调度）
        if self.hparams.lambda_schedule:
            p = self.current_iteration / self.max_iterations
            lambda_val = calc_coeff(self.current_iteration, max_iter=self.max_iterations)
            self.grl.set_lambda(lambda_val)
        else:
            lambda_val = self.hparams.lambda_domain
        
        self.current_iteration += 1
        
        # ============ 1. 特征提取和分类 ============
        # 源域
        feat_src, logits_src = self.get_features_and_logits(x_src)
        softmax_src = F.softmax(logits_src, dim=1)
        
        # 目标域
        feat_tgt, logits_tgt = self.get_features_and_logits(x_tgt)
        softmax_tgt = F.softmax(logits_tgt, dim=1)
        
        # ============ 2. 分类损失（仅源域） ============
        cls_loss = F.cross_entropy(logits_src, y_src)
        
        # ============ 3. CDAN 条件特征 ============
        if self.random_layer is not None:
            # 使用随机层降维
            cond_feat_src = self.random_layer(feat_src, softmax_src)
            cond_feat_tgt = self.random_layer(feat_tgt, softmax_tgt)
        else:
            # 直接计算外积
            cond_feat_src = cdan_multilinear_map(feat_src, softmax_src)
            cond_feat_tgt = cdan_multilinear_map(feat_tgt, softmax_tgt)
        
        # ============ 4. 梯度反转 + 域判别 ============
        cond_feat_src_grl = self.grl(cond_feat_src)
        cond_feat_tgt_grl = self.grl(cond_feat_tgt)
        
        domain_pred_src = self.discriminator(cond_feat_src_grl)
        domain_pred_tgt = self.discriminator(cond_feat_tgt_grl)
        
        # ============ 5. 域对抗损失 ============
        domain_loss = self.cdan_loss(
            domain_pred_src, domain_pred_tgt,
            softmax_src, softmax_tgt
        )
        
        # ============ 6. 熵最小化损失（可选） ============
        if self.hparams.lambda_entropy > 0:
            entropy_loss = entropy(softmax_tgt, reduction='mean')
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
        
        # ============ 7. 总损失 ============
        total_loss = (
            cls_loss + 
            self.hparams.lambda_domain * domain_loss + 
            self.hparams.lambda_entropy * entropy_loss
        )
        
        # ============ 8. 日志记录 ============
        # 计算准确率
        acc_src = accuracy(logits_src, y_src, task="multiclass", 
                          num_classes=self.hparams.n_classes)
        
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_loss", domain_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc_src, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lambda", lambda_val, prog_bar=False, on_step=False, on_epoch=True)
        
        if self.hparams.lambda_entropy > 0:
            self.log("train_entropy_loss", entropy_loss, prog_bar=False, on_step=False, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """验证步骤（使用目标域数据）"""
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """测试步骤"""
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
        """测试结束时计算混淆矩阵"""
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
        """训练开始时更新最大迭代次数估计"""
        if hasattr(self.trainer, 'num_training_batches'):
            self.max_iterations = self.hparams.max_epochs * self.trainer.num_training_batches
