import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import math
from typing import Dict, List, Tuple, Union, Optional
import gym
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class FinancialTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    金融数据专用的Transformer特征提取器
    
    将股票交易状态转换为序列形式，使用Transformer处理
    """
    def __init__(self, observation_space: gym.Space, 
                 d_model: int = 256,
                 nhead: int = 8, 
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_sequence_length: int = 100):
        
        # 计算特征维度
        features_dim = d_model
        super().__init__(observation_space, features_dim)
        
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # 输入投影层：将观测空间映射到d_model维度
        self.input_projection = nn.Linear(observation_space.shape[0], d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_sequence_length)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, features_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: shape (batch_size, obs_dim)
        Returns:
            features: shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # 输入投影 (batch_size, obs_dim) -> (batch_size, d_model)
        x = self.input_projection(observations)
        
        # 为了使用Transformer，我们需要序列维度
        # 这里我们将特征重塑为序列形式
        # 实际应用中，可以考虑历史时间步作为序列
        seq_len = min(self.max_sequence_length, x.shape[1])
        
        # 重塑为序列：(batch_size, d_model) -> (seq_len, batch_size, d_model)
        x = x.unsqueeze(0).repeat(seq_len, 1, 1)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 全局平均池化：(seq_len, batch_size, d_model) -> (batch_size, d_model)
        x = x.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 输出投影
        features = self.output_projection(x)
        
        return features

class TransformerActorCriticPolicy(BasePolicy):
    """
    基于Transformer的Actor-Critic策略网络
    
    使用Transformer作为特征提取器，支持连续动作空间
    """
    def __init__(self, 
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 lr_schedule,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn = nn.ReLU,
                 **kwargs):
        
        super().__init__(observation_space, action_space, **kwargs)
        
        # 动作空间检查
        assert isinstance(action_space, gym.spaces.Box), "只支持连续动作空间"
        
        # 默认网络架构
        if net_arch is None:
            net_arch = [dict(pi=[256, 256], vf=[256, 256])]
        
        # Transformer特征提取器
        self.features_extractor = FinancialTransformerFeatureExtractor(
            observation_space, d_model, nhead, num_layers, 
            dim_feedforward, dropout
        )
        
        # Actor网络（策略网络）
        self.action_dim = action_space.shape[0]
        actor_layers = []
        input_dim = self.features_extractor.features_dim
        
        # 根据net_arch构建Actor网络
        if isinstance(net_arch[0], dict) and 'pi' in net_arch[0]:
            actor_arch = net_arch[0]['pi']
        else:
            actor_arch = [256, 256]
            
        for hidden_dim in actor_arch:
            actor_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        # 输出层：均值和对数标准差
        actor_layers.append(nn.Linear(input_dim, self.action_dim))
        self.actor_mean = nn.Sequential(*actor_layers)
        
        # 动作标准差（可学习参数）
        self.action_log_std = nn.Parameter(
            torch.zeros(self.action_dim), requires_grad=True
        )
        
        # Critic网络（价值网络）
        critic_layers = []
        input_dim = self.features_extractor.features_dim
        
        if isinstance(net_arch[0], dict) and 'vf' in net_arch[0]:
            critic_arch = net_arch[0]['vf']
        else:
            critic_arch = [256, 256]
            
        for hidden_dim in critic_arch:
            critic_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # 动作分布
        self.action_dist = DiagGaussianDistribution(self.action_dim)
        
    def _get_constructor_parameters(self) -> Dict[str, any]:
        """返回构造函数参数，用于保存/加载模型"""
        data = super()._get_constructor_parameters()
        data.update(dict(
            d_model=self.features_extractor.d_model,
            # 添加其他需要保存的参数...
        ))
        return data
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测值
            deterministic: 是否使用确定性策略
            
        Returns:
            actions: 动作
            values: 状态价值
            log_probs: 动作对数概率
        """
        # 特征提取
        features = self.features_extractor(obs)
        
        # Actor输出
        action_mean = self.actor_mean(features)
        
        # 价值函数输出
        values = self.critic(features)
        
        # 动作分布
        action_std = torch.exp(self.action_log_std)
        distribution = Normal(action_mean, action_std)
        
        if deterministic:
            actions = action_mean
        else:
            actions = distribution.sample()
            
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        actions = torch.tanh(actions)  # 限制动作范围到[-1, 1]
        
        return actions, values.flatten(), log_probs
        
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定观测和动作的价值和概率
        
        Args:
            obs: 观测值
            actions: 动作
            
        Returns:
            values: 状态价值
            log_probs: 动作对数概率  
            entropy: 策略熵
        """
        features = self.features_extractor(obs)
        
        # Actor输出
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.action_log_std)
        distribution = Normal(action_mean, action_std)
        
        # 价值函数输出
        values = self.critic(features)
        
        # 反向tanh变换
        actions_unbounded = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        
        # 计算对数概率
        log_probs = distribution.log_prob(actions_unbounded).sum(dim=-1)
        
        # 计算熵
        entropy = distribution.entropy().sum(dim=-1)
        
        return values.flatten(), log_probs, entropy
        
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """预测状态价值"""
        features = self.features_extractor(obs)
        return self.critic(features).flatten()

# 时序Transformer策略（处理历史时间序列）
class TemporalTransformerPolicy(TransformerActorCriticPolicy):
    """
    时序Transformer策略：专门处理历史时间序列数据
    
    假设输入是历史时间步的状态序列
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 lr_schedule, sequence_length: int = 50, **kwargs):
        
        self.sequence_length = sequence_length
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        处理时序数据
        
        Args:
            obs: shape (batch_size, sequence_length * feature_dim) 或 (batch_size, feature_dim)
        """
        if len(obs.shape) == 2 and obs.shape[1] > self.sequence_length:
            # 重塑为时序格式
            batch_size = obs.shape[0]
            feature_dim = obs.shape[1] // self.sequence_length
            obs = obs.view(batch_size, self.sequence_length, feature_dim)
            
        return super().forward(obs, deterministic)

# 使用示例
def create_transformer_ppo_model():
    """创建使用Transformer策略的PPO模型示例"""
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    
    # 假设的股票交易环境
    # env = make_vec_env('StockTrading-v0', n_envs=1)
    
    # Transformer策略参数
    policy_kwargs = dict(
        d_model=512,           # Transformer维度
        nhead=8,               # 注意力头数  
        num_layers=6,          # Transformer层数
        dim_feedforward=2048,  # 前馈网络维度
        dropout=0.1,           # Dropout率
        net_arch=[dict(pi=[512, 256], vf=[512, 256])],  # 最终MLP层
        activation_fn=nn.GELU, # 激活函数
    )
    
    # 创建PPO模型
    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./transformer_ppo_tensorboard/"
    )
    
    return model

if __name__ == "__main__":
    print("Transformer策略网络定义完成！")
    print("主要特点：")
    print("1. 使用多头自注意力机制")
    print("2. 支持时序特征建模") 
    print("3. 可处理长序列依赖关系")
    print("4. 兼容Stable Baselines3接口") 