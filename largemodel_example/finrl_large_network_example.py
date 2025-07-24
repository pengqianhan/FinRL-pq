"""
FinRL中使用大型神经网络的完整示例

展示如何将MlpPolicy替换为Transformer、ResNet等大型神经网络
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# 导入FinRL相关模块
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS

# 导入我们定义的大型网络策略
from largemodel.transformer_policy_example import TransformerActorCriticPolicy
from largemodel.advanced_policies_example import (
    LargeNeuralNetworkPolicy, 
    CNNLSTMFeatureExtractor,
    ResNetFeatureExtractor,
    ViTStyleFeatureExtractor
)

class FinRLDRLAgentLarge:
    """
    支持大型神经网络的FinRL DRL智能体
    
    扩展原有的DRLAgent类，支持自定义策略网络
    """
    
    def __init__(self, env):
        self.env = env
    
    def get_large_model(self, 
                       model_name: str = "ppo",
                       policy_class=None,
                       policy_kwargs=None,
                       model_kwargs=None,
                       verbose=1):
        """
        创建使用大型神经网络的模型
        
        Args:
            model_name: 算法名称 ('ppo', 'sac', 等)
            policy_class: 自定义策略类
            policy_kwargs: 策略网络参数
            model_kwargs: 算法超参数
        """
        
        # 默认策略参数
        if policy_kwargs is None:
            policy_kwargs = {}
            
        # 默认算法参数（针对大型网络优化）
        if model_kwargs is None:
            model_kwargs = {
                'learning_rate': 1e-4,      # 降低学习率
                'n_steps': 4096,            # 增加收集步数
                'batch_size': 32,           # 减少批次大小
                'n_epochs': 5,              # 减少更新轮数
                'gamma': 0.999,             # 更高折扣因子
                'gae_lambda': 0.98,
                'clip_range': 0.1,          # 更小裁剪范围
                'ent_coef': 0.001,          # 降低熵系数
                'vf_coef': 0.5,
                'max_grad_norm': 1.0,       # 梯度裁剪
            }
        
        # 支持的算法映射
        ALGORITHMS = {
            'ppo': PPO,
            # 可以添加其他算法...
        }
        
        if model_name not in ALGORITHMS:
            raise ValueError(f"不支持的算法: {model_name}")
            
        algorithm_class = ALGORITHMS[model_name]
        
        # 创建模型
        if policy_class is not None:
            # 使用自定义策略类
            model = algorithm_class(
                policy=policy_class,
                env=self.env,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                **model_kwargs
            )
        else:
            # 使用默认策略
            model = algorithm_class(
                policy="MlpPolicy",
                env=self.env,
                verbose=verbose,
                **model_kwargs
            )
            
        return model
    
    def train_model(self, model, tb_log_name, total_timesteps=200000):
        """训练模型（与原DRLAgent保持一致）"""
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name
        )
        return model

# 训练监控回调
class LargeModelTrainingCallback(BaseCallback):
    """大型模型训练监控回调"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # 记录训练指标
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        
        # 每1000步打印一次统计信息
        if self.num_timesteps % 1000 == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                self.logger.record("train/mean_episode_reward", mean_reward)
                
        return True

def create_stock_trading_environment(train_data):
    """
    创建股票交易环境
    
    Args:
        train_data: 训练数据DataFrame
    """
    
    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    return env_train, env_kwargs

def example_1_transformer_policy():
    """
    示例1: 使用Transformer策略进行股票交易
    """
    print("=" * 60)
    print("示例1: Transformer策略网络")
    print("=" * 60)
    
    # 假设已有训练数据
    # train_data = pd.read_csv('your_processed_stock_data.csv')
    # env_train, _ = create_stock_trading_environment(train_data)
    
    # 创建模拟环境用于演示
    print("正在创建Transformer策略...")
    
    # Transformer策略参数
    transformer_policy_kwargs = {
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'net_arch': [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])],
        'activation_fn': nn.GELU
    }
    
    # 算法超参数（针对大型网络优化）
    ppo_params_transformer = {
        'learning_rate': 5e-5,      # 更低的学习率
        'n_steps': 8192,            # 更多样本收集
        'batch_size': 64,           # 适中的批次大小
        'n_epochs': 3,              # 较少的更新轮数
        'gamma': 0.999,
        'gae_lambda': 0.98,
        'clip_range': 0.1,
        'ent_coef': 0.0005,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }
    
    print("Transformer策略配置:")
    print(f"- 模型维度: {transformer_policy_kwargs['d_model']}")
    print(f"- 注意力头数: {transformer_policy_kwargs['nhead']}")
    print(f"- Transformer层数: {transformer_policy_kwargs['num_layers']}")
    print(f"- 学习率: {ppo_params_transformer['learning_rate']}")
    
    # 创建智能体（演示用，实际使用时需要真实环境）
    # agent = FinRLDRLAgentLarge(env=env_train)
    # model = agent.get_large_model(
    #     model_name="ppo",
    #     policy_class=TransformerActorCriticPolicy,
    #     policy_kwargs=transformer_policy_kwargs,
    #     model_kwargs=ppo_params_transformer
    # )
    
    # # 设置日志
    # logger = configure("./logs/transformer_ppo", ["stdout", "csv", "tensorboard"])
    # model.set_logger(logger)
    
    # # 训练
    # trained_model = agent.train_model(
    #     model=model,
    #     tb_log_name="transformer_stock_trading",
    #     total_timesteps=500000
    # )
    
    print("✅ Transformer策略配置完成！")
    return transformer_policy_kwargs, ppo_params_transformer

def example_2_cnn_lstm_policy():
    """
    示例2: 使用CNN-LSTM策略处理时序数据
    """
    print("\n" + "=" * 60)
    print("示例2: CNN-LSTM策略网络")
    print("=" * 60)
    
    # CNN-LSTM策略参数
    cnn_lstm_policy_kwargs = {
        'feature_extractor_class': CNNLSTMFeatureExtractor,
        'feature_extractor_kwargs': {
            'cnn_channels': [64, 128, 256, 512],
            'lstm_hidden_size': 512,
            'lstm_num_layers': 3,
            'sequence_length': 50,
            'features_dim': 1024
        },
        'net_arch': [dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])],
        'activation_fn': nn.GELU
    }
    
    # 针对时序数据的训练参数
    ppo_params_cnn_lstm = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 32,           # 由于LSTM计算复杂，使用较小批次
        'n_epochs': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 1.0,
    }
    
    print("CNN-LSTM策略配置:")
    print(f"- CNN通道: {cnn_lstm_policy_kwargs['feature_extractor_kwargs']['cnn_channels']}")
    print(f"- LSTM隐藏维度: {cnn_lstm_policy_kwargs['feature_extractor_kwargs']['lstm_hidden_size']}")
    print(f"- LSTM层数: {cnn_lstm_policy_kwargs['feature_extractor_kwargs']['lstm_num_layers']}")
    print(f"- 序列长度: {cnn_lstm_policy_kwargs['feature_extractor_kwargs']['sequence_length']}")
    
    print("✅ CNN-LSTM策略配置完成！")
    return cnn_lstm_policy_kwargs, ppo_params_cnn_lstm

def example_3_resnet_policy():
    """
    示例3: 使用ResNet策略网络
    """
    print("\n" + "=" * 60)
    print("示例3: ResNet策略网络")
    print("=" * 60)
    
    # ResNet策略参数
    resnet_policy_kwargs = {
        'feature_extractor_class': ResNetFeatureExtractor,
        'feature_extractor_kwargs': {
            'hidden_dim': 1024,
            'num_blocks': 16,           # 深度残差网络
            'features_dim': 1024
        },
        'net_arch': [dict(pi=[2048, 1024, 512, 256], vf=[2048, 1024, 512, 256])],
        'activation_fn': nn.GELU
    }
    
    # ResNet训练参数
    ppo_params_resnet = {
        'learning_rate': 1e-4,
        'n_steps': 4096,
        'batch_size': 64,
        'n_epochs': 5,
        'gamma': 0.999,
        'gae_lambda': 0.98,
        'clip_range': 0.15,
        'ent_coef': 0.001,
        'vf_coef': 0.5,
        'max_grad_norm': 0.8,
    }
    
    print("ResNet策略配置:")
    print(f"- 隐藏维度: {resnet_policy_kwargs['feature_extractor_kwargs']['hidden_dim']}")
    print(f"- 残差块数量: {resnet_policy_kwargs['feature_extractor_kwargs']['num_blocks']}")
    print(f"- 特征维度: {resnet_policy_kwargs['feature_extractor_kwargs']['features_dim']}")
    
    print("✅ ResNet策略配置完成！")
    return resnet_policy_kwargs, ppo_params_resnet

def example_4_vit_policy():
    """
    示例4: 使用Vision Transformer风格策略
    """
    print("\n" + "=" * 60)
    print("示例4: Vision Transformer策略网络")
    print("=" * 60)
    
    # ViT策略参数
    vit_policy_kwargs = {
        'feature_extractor_class': ViTStyleFeatureExtractor,
        'feature_extractor_kwargs': {
            'd_model': 768,             # 类似ViT-Base
            'num_heads': 12,
            'num_layers': 12,
            'patch_size': 16,
            'features_dim': 1024
        },
        'net_arch': [dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])],
        'activation_fn': nn.GELU
    }
    
    # ViT训练参数
    ppo_params_vit = {
        'learning_rate': 3e-5,          # 非常低的学习率
        'n_steps': 8192,                # 大量样本收集
        'batch_size': 16,               # 小批次以适应GPU内存
        'n_epochs': 3,                  # 少量更新轮数
        'gamma': 0.999,
        'gae_lambda': 0.98,
        'clip_range': 0.1,
        'ent_coef': 0.0001,
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,
    }
    
    print("ViT策略配置:")
    print(f"- 模型维度: {vit_policy_kwargs['feature_extractor_kwargs']['d_model']}")
    print(f"- 注意力头数: {vit_policy_kwargs['feature_extractor_kwargs']['num_heads']}")
    print(f"- Transformer层数: {vit_policy_kwargs['feature_extractor_kwargs']['num_layers']}")
    print(f"- Patch大小: {vit_policy_kwargs['feature_extractor_kwargs']['patch_size']}")
    
    print("✅ ViT策略配置完成！")
    return vit_policy_kwargs, ppo_params_vit

def compare_model_complexities():
    """
    对比不同模型的复杂度
    """
    print("\n" + "=" * 60)
    print("模型复杂度对比")
    print("=" * 60)
    
    models_info = [
        {
            'name': 'MlpPolicy (原始)',
            'parameters': '~200K',
            'memory': '~100MB',
            'training_time': '快速',
            'performance': '基准'
        },
        {
            'name': 'Transformer',
            'parameters': '~10M-50M',
            'memory': '~500MB-2GB',
            'training_time': '中等',
            'performance': '强于MLP，适合复杂模式'
        },
        {
            'name': 'CNN-LSTM',
            'parameters': '~2M-10M',
            'memory': '~300MB-1GB',
            'training_time': '中等',
            'performance': '善于时序建模'
        },
        {
            'name': 'ResNet-16',
            'parameters': '~5M-20M',
            'memory': '~400MB-1.5GB',
            'training_time': '中等',
            'performance': '深度网络，梯度稳定'
        },
        {
            'name': 'ViT-Base',
            'parameters': '~20M-100M',
            'memory': '~1GB-4GB',
            'training_time': '慢',
            'performance': '最强表达能力'
        }
    ]
    
    print(f"{'模型':<15} {'参数量':<12} {'显存占用':<12} {'训练速度':<10} {'性能特点'}")
    print("-" * 70)
    
    for model in models_info:
        print(f"{model['name']:<15} {model['parameters']:<12} {model['memory']:<12} "
              f"{model['training_time']:<10} {model['performance']}")

def training_recommendations():
    """
    大型网络训练建议
    """
    print("\n" + "=" * 60)
    print("大型网络训练建议")
    print("=" * 60)
    
    recommendations = [
        "1. 硬件要求:",
        "   - GPU: 至少8GB显存 (推荐RTX 3080/4080或以上)",
        "   - 内存: 32GB以上",
        "   - 存储: SSD存储，快速数据加载",
        "",
        "2. 训练策略:",
        "   - 使用混合精度训练 (torch.cuda.amp)",
        "   - 梯度累积以模拟大批次",
        "   - 学习率预热和衰减调度",
        "   - 定期保存检查点",
        "",
        "3. 超参数调优:",
        "   - 从小学习率开始 (1e-5到1e-4)",
        "   - 增加样本收集步数 (4096-8192)",
        "   - 减少批次大小适应GPU内存",
        "   - 使用更强的正则化 (dropout, weight decay)",
        "",
        "4. 监控指标:",
        "   - 训练损失和验证损失",
        "   - 梯度范数和参数更新幅度",
        "   - GPU内存使用率",
        "   - 每步训练时间",
        "",
        "5. 调试技巧:",
        "   - 从小数据集开始验证",
        "   - 可视化注意力权重 (Transformer)",
        "   - 监控网络激活分布",
        "   - 使用TensorBoard记录详细信息"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """
    主函数：运行所有示例
    """
    print("🚀 FinRL大型神经网络策略替换示例")
    print("🎯 展示如何将MlpPolicy替换为Transformer等大型网络")
    
    # 运行各个示例
    example_1_transformer_policy()
    example_2_cnn_lstm_policy() 
    example_3_resnet_policy()
    example_4_vit_policy()
    
    # 对比和建议
    compare_model_complexities()
    training_recommendations()
    
    print("\n" + "=" * 60)
    print("🎉 所有示例运行完成！")
    print("💡 要实际运行，请:")
    print("   1. 准备股票数据")
    print("   2. 创建交易环境") 
    print("   3. 选择合适的策略网络")
    print("   4. 配置训练参数")
    print("   5. 开始训练和测试")
    print("=" * 60)

if __name__ == "__main__":
    main() 