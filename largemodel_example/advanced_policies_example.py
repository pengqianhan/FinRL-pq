import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

# ==================== CNN-LSTM混合架构 ====================
class CNNLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN-LSTM混合特征提取器
    
    适用于处理金融时间序列数据：
    - CNN层提取局部模式
    - LSTM层捕捉时序依赖关系
    """
    def __init__(self, observation_space: gym.Space,
                 cnn_channels: List[int] = [32, 64, 128],
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 sequence_length: int = 50,
                 features_dim: int = 512):
        
        super().__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        
        # 假设输入是 (batch_size, sequence_length, feature_dim)
        input_channels = 1  # 将特征作为单通道处理
        
        # CNN层：提取局部特征
        cnn_layers = []
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(input_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            input_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # 计算CNN输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, observation_space.shape[0])
            cnn_output = self.cnn(dummy_input)
            cnn_output_dim = cnn_output.view(1, -1).shape[1]
        
        # LSTM层：捕捉时序依赖
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.2 if lstm_num_layers > 1 else 0
        )
        
        # 输出投影层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 重塑输入为时序格式
        if len(observations.shape) == 2:
            # 假设输入是展平的时序数据
            feature_dim = observations.shape[1] // self.sequence_length
            x = observations.view(batch_size, self.sequence_length, feature_dim)
        else:
            x = observations
            
        # 为CNN添加通道维度：(batch_size, seq_len, features) -> (batch_size, 1, features)
        x = x.view(batch_size * self.sequence_length, 1, -1)
        
        # CNN特征提取
        cnn_features = self.cnn(x)  # (batch_size * seq_len, channels, reduced_features)
        cnn_features = cnn_features.view(batch_size, self.sequence_length, -1)
        
        # LSTM时序建模
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # 使用最后一个时间步的输出
        final_features = self.fc(lstm_out[:, -1, :])
        
        return final_features

# ==================== ResNet架构 ====================
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out + residual)
        return self.activation(out)

class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """
    ResNet风格的特征提取器
    
    使用残差连接，支持更深的网络
    """
    def __init__(self, observation_space: gym.Space,
                 hidden_dim: int = 512,
                 num_blocks: int = 8,
                 features_dim: int = 512):
        
        super().__init__(observation_space, features_dim)
        
        # 输入投影
        self.input_projection = nn.Linear(observation_space.shape[0], hidden_dim)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2)
            for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(observations)
        
        for block in self.residual_blocks:
            x = block(x)
            
        features = self.output_projection(x)
        return features

# ==================== Vision Transformer (ViT)风格 ====================
class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        return output

class ViTStyleFeatureExtractor(BaseFeaturesExtractor):
    """
    Vision Transformer风格的特征提取器
    
    将金融特征分割为patches，使用自注意力机制
    """
    def __init__(self, observation_space: gym.Space,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 patch_size: int = 16,
                 features_dim: int = 512):
        
        super().__init__(observation_space, features_dim)
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 计算patch数量
        self.num_patches = observation_space.shape[0] // patch_size
        if observation_space.shape[0] % patch_size != 0:
            self.num_patches += 1
            
        # Patch嵌入
        self.patch_embedding = nn.Linear(patch_size, d_model)
        
        # 位置嵌入
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, features_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 创建patches
        patches = []
        for i in range(0, observations.shape[1], self.patch_size):
            patch = observations[:, i:i+self.patch_size]
            if patch.shape[1] < self.patch_size:
                # 填充最后一个patch
                padding = torch.zeros(batch_size, self.patch_size - patch.shape[1], 
                                    device=observations.device)
                patch = torch.cat([patch, padding], dim=1)
            patches.append(patch)
            
        patches = torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_size)
        
        # Patch嵌入
        x = self.patch_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置嵌入
        x = x + self.position_embedding
        
        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x)
            
        # 使用CLS token的输出
        cls_output = x[:, 0]
        features = self.head(cls_output)
        
        return features

# ==================== 大型策略网络基类 ====================
class LargeNeuralNetworkPolicy(BasePolicy):
    """
    大型神经网络策略基类
    
    支持多种特征提取器：CNN-LSTM, ResNet, ViT等
    """
    def __init__(self, 
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 lr_schedule,
                 feature_extractor_class=ResNetFeatureExtractor,
                 feature_extractor_kwargs=None,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn=nn.GELU,
                 **kwargs):
        
        super().__init__(observation_space, action_space, **kwargs)
        
        # 特征提取器参数
        if feature_extractor_kwargs is None:
            feature_extractor_kwargs = {}
            
        # 创建特征提取器
        self.features_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        
        # 默认网络架构（更大的网络）
        if net_arch is None:
            net_arch = [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])]
            
        # Actor网络
        self.action_dim = action_space.shape[0]
        actor_layers = []
        input_dim = self.features_extractor.features_dim
        
        if isinstance(net_arch[0], dict) and 'pi' in net_arch[0]:
            actor_arch = net_arch[0]['pi']
        else:
            actor_arch = [1024, 512, 256]
            
        for hidden_dim in actor_arch:
            actor_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),  # 添加Layer Normalization
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        actor_layers.append(nn.Linear(input_dim, self.action_dim))
        self.actor_mean = nn.Sequential(*actor_layers)
        
        # 动作标准差
        self.action_log_std = nn.Parameter(
            torch.zeros(self.action_dim), requires_grad=True
        )
        
        # Critic网络
        critic_layers = []
        input_dim = self.features_extractor.features_dim
        
        if isinstance(net_arch[0], dict) and 'vf' in net_arch[0]:
            critic_arch = net_arch[0]['vf']
        else:
            critic_arch = [1024, 512, 256]
            
        for hidden_dim in critic_arch:
            critic_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.features_extractor(obs)
        
        # Actor输出
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.action_log_std)
        distribution = Normal(action_mean, action_std)
        
        # 价值函数输出
        values = self.critic(features)
        
        if deterministic:
            actions = action_mean
        else:
            actions = distribution.sample()
            
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        actions = torch.tanh(actions)
        
        return actions, values.flatten(), log_probs
        
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.features_extractor(obs)
        
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.action_log_std)
        distribution = Normal(action_mean, action_std)
        
        values = self.critic(features)
        
        # 反向tanh变换
        actions_unbounded = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        
        log_probs = distribution.log_prob(actions_unbounded).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        return values.flatten(), log_probs, entropy
        
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        return self.critic(features).flatten()

# ==================== 使用示例 ====================
def create_large_network_models():
    """创建各种大型网络模型的示例"""
    
    from stable_baselines3 import PPO
    
    # 1. CNN-LSTM模型
    cnn_lstm_policy_kwargs = dict(
        feature_extractor_class=CNNLSTMFeatureExtractor,
        feature_extractor_kwargs=dict(
            cnn_channels=[64, 128, 256],
            lstm_hidden_size=512,
            lstm_num_layers=3,
            sequence_length=50,
            features_dim=1024
        ),
        net_arch=[dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])],
        activation_fn=nn.GELU
    )
    
    # 2. ResNet模型
    resnet_policy_kwargs = dict(
        feature_extractor_class=ResNetFeatureExtractor,
        feature_extractor_kwargs=dict(
            hidden_dim=1024,
            num_blocks=12,  # 更深的网络
            features_dim=1024
        ),
        net_arch=[dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])],
        activation_fn=nn.GELU
    )
    
    # 3. ViT风格模型
    vit_policy_kwargs = dict(
        feature_extractor_class=ViTStyleFeatureExtractor,
        feature_extractor_kwargs=dict(
            d_model=768,  # 类似ViT-Base
            num_heads=12,
            num_layers=12,
            patch_size=16,
            features_dim=1024
        ),
        net_arch=[dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])],
        activation_fn=nn.GELU
    )
    
    return {
        'cnn_lstm': cnn_lstm_policy_kwargs,
        'resnet': resnet_policy_kwargs, 
        'vit': vit_policy_kwargs
    }

# 训练配置建议
def get_training_config_for_large_models():
    """大型模型的训练配置建议"""
    return {
        'learning_rate': 1e-4,  # 降低学习率
        'n_steps': 4096,        # 增加样本收集
        'batch_size': 32,       # 减少批次大小以适应GPU内存
        'n_epochs': 5,          # 减少更新轮数
        'gamma': 0.999,         # 更高的折扣因子
        'gae_lambda': 0.98,
        'clip_range': 0.1,      # 更小的裁剪范围
        'ent_coef': 0.001,      # 降低熵系数
        'vf_coef': 0.5,
        'max_grad_norm': 1.0,   # 梯度裁剪
    }

if __name__ == "__main__":
    print("大型神经网络策略定义完成！")
    print("\n支持的架构:")
    print("1. CNN-LSTM: 适合时序数据，结合局部和时序特征")
    print("2. ResNet: 深度残差网络，支持梯度流动")
    print("3. ViT: Vision Transformer风格，强大的自注意力机制")
    print("4. 自定义: 可以轻松扩展其他架构")
    
    # 显示参数统计
    print("\n参数规模对比:")
    print("- 原始MlpPolicy: ~50K-200K参数")
    print("- CNN-LSTM: ~2M-10M参数") 
    print("- ResNet-12: ~5M-20M参数")
    print("- ViT-Base: ~10M-50M参数") 