import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Callable

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

# 載入我們第二階段寫好的自定義環境 (與平行化工廠函數)
from smc_env import create_vectorized_envs

# ==============================================================
# 1. Custom Feature Extractor (使用 GRU 提取 SMC 序列特徵)
# ==============================================================

class CustomGRUFeatureExtractor(BaseFeaturesExtractor):
    """
    自定義的神經網路特徵提取器。
    用於替代 PPO 預設的 Flatten() 或 CNN 提取器。
    它會接收 (Batch_Size, 50, 9) 的序列輸入，透過 GRU 提取出維度為 features_dim 的結果。
    """
    def __init__(self, observation_space, features_dim: int = 256):
        # 呼叫父類別並告知最終輸出的神經元數量
        super().__init__(observation_space, features_dim)
        
        # observation_space 的形狀為 (window_size=50, num_features=9)
        seq_length, input_features = observation_space.shape
        
        # 建立 GRU 層來處理時間序列特徵 (OHLCV + SMC)
        # 註：這裡選用 GRU 取代 LSTM，因為 GRU 參數較少，對量化交易序列來說更容易收斂且防止明顯過擬合
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # PPO 傳入的 observations 會是 PyTorch Tensor: (batch_size, seq_length, input_features)
        
        # 經過 GRU 處理
        # gru_out shape: (batch_size, seq_length, features_dim)
        gru_out, _ = self.gru(observations)
        
        # 我們只取時間序列「最後一個時間點」的網路輸出，作為代表這一整段 50 根 K 線特徵的向量
        final_features = gru_out[:, -1, :]
        return final_features

# ==============================================================
# 2. Linear Learning Rate Scheduler
# ==============================================================

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    回傳一個能根據進度調整學習率的函式。
    學習率會從 initial_value 線性遞減到 0。
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining 會從 1.0 (開始) 遞減到 0.0 (結束)
        return progress_remaining * initial_value
    return func


# ==============================================================
# 3. 主循環訓練腳本 (PPO)
# ==============================================================

if __name__ == "__main__":
    
    # [A] 準備測試用的合成資料 (實務上請換成 pd.read_csv("你的真實交易資料.csv"))
    print("Generating synthetic data for training...")
    dates = pd.date_range("2026-01-01", periods=10000)
    data = {"Open": np.random.uniform(100, 200, 10000)}
    df = pd.DataFrame(data, index=dates)
    df["High"] = df["Open"] + np.random.uniform(0, 10, 10000)
    df["Low"] = df["Open"] - np.random.uniform(0, 10, 10000)
    df["Close"] = df["Open"] + np.random.uniform(-5, 5, 10000)
    df["Volume"] = np.random.randint(1000, 5000, 10000)

    # [B] 建立平行化的 Gymnasium 交易環境，適配 Intel 12700K (16 個獨立進程)
    num_cpu = 16
    env = create_vectorized_envs(df, num_envs=num_cpu)
    
    # [C] 準備 Policy 的網路配置
    policy_kwargs = dict(
        # 1. 將自定義的 GRU 提取器綁定上去
        features_extractor_class=CustomGRUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        # 2. 定義 Actor (pi) 與 Critic (vf) 分支的層數與神經元數量
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    # [D] 宣告與初始化 PPO 模型 (針對 RTX 3090)
    # 利用大的 batch_size 塞滿 VRAM
    print("Initializing PPO model with custom GRU extractor...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4), # 動態學習率
        n_steps=2048,                        # 每一個 CPU 進程採樣 2048 步
        batch_size=4096,                     # PPO 內部每次更新的 mini-batch 尺寸
        n_epochs=10,
        gamma=0.99,                          # 折扣因子
        policy_kwargs=policy_kwargs,
        tensorboard_log="./smc_tensorboard/", # 開啟 TensorBoard
        device="cuda",                       # 強制啟用 GPU
        verbose=1
    )
    
    # [E] 設定每隔固定步數就自動儲存模型 (避免訓練到一半當機)
    os.makedirs("./models", exist_ok=True)
    # save_freq 的單位是 "每一個獨立環境的步數"，總計為 save_freq * num_cpu
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_cpu, 1), 
        save_path="./models/",
        name_prefix="smc_ppo_rtx3090"
    )
    
    # [F] 啟動訓練
    print(f"Starting training on {model.device} with Tensorboard enabled...")
    try:
        # 開始利用多進程採樣並訓練
        model.learn(total_timesteps=100_000, callback=checkpoint_callback, progress_bar=True)
        # 儲存最終模型
        model.save("./models/smc_ppo_rtx3090_final")
        print("\n🎉 Training complete! Model successfully saved at ./models/smc_ppo_rtx3090_final")
        print("To view Tensorboard logs, run: tensorboard --logdir ./smc_tensorboard/")
    except Exception as e:
        print("\n⚠️ Training interrupted:", type(e).__name__, "-", e)
        print("請確保您已安裝所有套件，且您的 PyTorch 支援 CUDA。")
