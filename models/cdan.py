"""
CDAN (Conditional Domain Adversarial Network) 组件

包含:
- GradientReversalLayer: 梯度反转层
- DomainDiscriminator: 域判别器
- RandomLayer: 随机投影层（用于降维）
- CDAN 条件特征生成函数
"""

import torch
from torch import nn
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转函数
    前向传播时直接传递输入，反向传播时将梯度乘以 -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        # 反向传播时反转梯度
        return -ctx.lambda_ * grads, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)
    
    在前向传播时作为恒等映射，在反向传播时将梯度乘以 -lambda
    用于对抗训练中，使特征提取器学习域不变特征
    
    Args:
        lambda_: 梯度反转系数，默认为 1.0
    """
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """动态设置 lambda 值"""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    域判别器
    
    用于区分特征来自源域还是目标域
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度，默认为 256
        num_layers: 隐藏层数量，默认为 2
        dropout: Dropout 比率，默认为 0.5
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.5
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # 最后一层输出单个值（二分类：源域 vs 目标域）
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.net(x)


class RandomLayer(nn.Module):
    """
    随机投影层
    
    当 CDAN 的条件特征维度过大时（d_model * n_classes），
    使用随机矩阵将其投影到较低维度
    
    Args:
        input_dims: 输入维度列表 [特征维度, 类别数]
        output_dim: 输出维度，默认为 1024
    """
    def __init__(self, input_dims: list, output_dim: int = 1024):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # 创建随机矩阵（不参与训练）
        self.register_buffer(
            'random_matrix',
            torch.randn(np.prod(input_dims), output_dim)
        )
    
    def forward(self, features, softmax_output):
        """
        Args:
            features: [B, d_model] 特征
            softmax_output: [B, n_classes] 分类器 softmax 输出
        Returns:
            [B, output_dim] 降维后的条件特征
        """
        # 计算外积
        batch_size = features.size(0)
        # [B, d_model, 1] x [B, 1, n_classes] -> [B, d_model, n_classes]
        outer_product = torch.bmm(features.unsqueeze(2), softmax_output.unsqueeze(1))
        # 展平为 [B, d_model * n_classes]
        outer_product = outer_product.view(batch_size, -1)
        # 随机投影
        output = torch.mm(outer_product, self.random_matrix)
        return output


def cdan_multilinear_map(features: torch.Tensor, softmax_output: torch.Tensor) -> torch.Tensor:
    """
    CDAN 的核心：计算特征与分类预测的外积（多线性映射）
    
    这是 CDAN 相比普通 DANN 的关键改进：
    - DANN 只使用特征进行域对抗
    - CDAN 使用特征和分类预测的联合分布进行域对抗
    
    Args:
        features: [B, d_model] 特征提取器输出的特征
        softmax_output: [B, n_classes] 分类器的 softmax 输出
    
    Returns:
        [B, d_model * n_classes] 条件特征
    """
    batch_size = features.size(0)
    # [B, d_model, 1] x [B, 1, n_classes] -> [B, d_model, n_classes]
    outer_product = torch.bmm(features.unsqueeze(2), softmax_output.unsqueeze(1))
    # 展平为 [B, d_model * n_classes]
    return outer_product.view(batch_size, -1)


def entropy(predictions: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    计算预测的熵
    
    用于熵最小化正则化，鼓励模型对目标域样本做出更自信的预测
    
    Args:
        predictions: [B, n_classes] softmax 预测概率
        reduction: 'mean', 'sum', 或 'none'
    
    Returns:
        熵值
    """
    epsilon = 1e-8
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    
    if reduction == 'mean':
        return H.mean()
    elif reduction == 'sum':
        return H.sum()
    else:
        return H


def calc_coeff(iter_num: int, high: float = 1.0, low: float = 0.0, 
               alpha: float = 10.0, max_iter: float = 10000.0) -> float:
    """
    计算 lambda 系数的调度函数
    
    使用 sigmoid 函数使 lambda 从 low 逐渐增加到 high
    这种渐进式增加有助于训练稳定性
    
    Args:
        iter_num: 当前迭代次数
        high: lambda 的最大值
        low: lambda 的最小值
        alpha: 控制增长速度的参数
        max_iter: 最大迭代次数
    
    Returns:
        当前的 lambda 值
    """
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


class CDANLoss(nn.Module):
    """
    CDAN 损失函数
    
    结合了：
    1. 分类损失（源域）
    2. 域对抗损失（源域 + 目标域）
    3. 可选的熵最小化损失（目标域）
    
    Args:
        use_entropy: 是否使用熵条件对抗（CDAN+E）
        use_random_layer: 是否使用随机层降维
        d_model: 特征维度
        n_classes: 类别数
        random_dim: 随机层输出维度
    """
    def __init__(
            self,
            use_entropy: bool = True,
            use_random_layer: bool = False,
            d_model: int = 32,
            n_classes: int = 4,
            random_dim: int = 1024
    ):
        super().__init__()
        self.use_entropy = use_entropy
        self.use_random_layer = use_random_layer
        
        if use_random_layer:
            self.random_layer = RandomLayer([d_model, n_classes], random_dim)
        else:
            self.random_layer = None
    
    def forward(
            self,
            domain_pred_src: torch.Tensor,
            domain_pred_tgt: torch.Tensor,
            softmax_src: torch.Tensor = None,
            softmax_tgt: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算域对抗损失
        
        Args:
            domain_pred_src: [B_src, 1] 源域的域判别器输出
            domain_pred_tgt: [B_tgt, 1] 目标域的域判别器输出
            softmax_src: [B_src, n_classes] 源域的 softmax 输出（用于熵加权）
            softmax_tgt: [B_tgt, n_classes] 目标域的 softmax 输出（用于熵加权）
        
        Returns:
            域对抗损失
        """
        # 域标签：源域为 0，目标域为 1
        device = domain_pred_src.device
        domain_label_src = torch.zeros(domain_pred_src.size(0), 1, device=device)
        domain_label_tgt = torch.ones(domain_pred_tgt.size(0), 1, device=device)
        
        if self.use_entropy and softmax_src is not None and softmax_tgt is not None:
            # CDAN+E: 使用熵加权
            # 熵越高（预测越不确定），权重越低
            entropy_src = entropy(softmax_src, reduction='none')
            entropy_tgt = entropy(softmax_tgt, reduction='none')
            
            # 归一化熵作为权重（需要 detach 以避免梯度问题）
            weight_src = (1.0 + torch.exp(-entropy_src)).detach()
            weight_tgt = (1.0 + torch.exp(-entropy_tgt)).detach()
            
            # 加权的二元交叉熵损失
            loss_src = nn.functional.binary_cross_entropy_with_logits(
                domain_pred_src, domain_label_src, weight=weight_src.unsqueeze(1)
            )
            loss_tgt = nn.functional.binary_cross_entropy_with_logits(
                domain_pred_tgt, domain_label_tgt, weight=weight_tgt.unsqueeze(1)
            )
        else:
            # 标准 CDAN
            loss_src = nn.functional.binary_cross_entropy_with_logits(
                domain_pred_src, domain_label_src
            )
            loss_tgt = nn.functional.binary_cross_entropy_with_logits(
                domain_pred_tgt, domain_label_tgt
            )
        
        return loss_src + loss_tgt
