# SMC-DRL: 智慧資金概念與深度強化學習交易系統

這是一個結合 **「智慧資金概念」(Smart Money Concept, SMC)** 與 **「深度強化學習」(Deep Reinforcement Learning, DRL)** 的自動化量化交易框架，專為現代高頻與波段交易設計。

本專案利用 PyTorch 與 Stable-Baselines3 實作，並針對多核心 CPU (如 Intel 12700K) 與高階 GPU (如 NVIDIA RTX 3090) 的硬體架構進行了平行化採樣與效能最佳化。

## 系統架構

此系統分為三個核心實作階段：

### 1. SMC 特徵提取器 (`smc_features.py`)
使用高度向量化的 Pandas 矩陣運算，快速從 OHLCV K 線資料中提取 SMC 關鍵交易特徵，解決了傳統跌代運算緩慢的問題：
- **FVG (Fair Value Gap)**：精準定位三根 K 線間的流動性失衡缺口。
- **Fractals (局部高低點)**：辨識市場的波段反折點。
- **MSS (Market Structure Shift)**：當確認後的 Fractal 被突破時發出訊號，並嚴格防範「未來函數 (Look-ahead bias)」干擾。

### 2. DRL 交易環境 (`smc_env.py`)
基於 Gymnasium 框架打造的客製化強化學習環境：
- **狀態/觀察空間 (State Space)**：包含 50 根最新 K 線視窗，匯入價格、FVG 寬度與多時框 RSI 特徵，並做完常態化處理。
- **動作空間 (Action Space)**：採用連續空間 `[-1, 1]`，模型能輸出任意數字來決定作空到作多的全倉與半倉倉位。
- **SMC 特化獎勵 (Shaping Reward)**：若模型懂得在 FVG 支撐壓力區間內執行順勢建倉，環境將給予額外的學習獎賞，促使模型強暴式地學會訂單流邏輯。同時結合了 **Max Drawdown (最大回撤) 懲罰機制** 來嚴格控管風險。

### 3. PPO 模型訓練腳本 (`train_ppo.py`)
為 RTX 3090 深度學習顯卡量身打造的高速訓練腳本：
- **自定義 GRU 提取層**：將 `(50, 9)` 的時間序列特徵送入自製的 PyTorch GRU 神經網路，完美捕捉金融市場的時間依賴性與長記憶特徵。
- **極致平行採樣**：透過 `SubprocVecEnv` 開啟最高 16 個子進程，全面汲取跨核心算力。
- **進階訓練設定**：加入了基於 TensorBoard 的即時監控、線性遞減的預測學習率 (LR Scheduler) 與極大 Batch Size 調校。

## 快速開始

### 環境安裝
請確保你的電腦具備支援 CUDA 的 PyTorch 環境：
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3 tensorboard gymnasium pandas numpy
```

### 啟動模型訓練
執行訓練腳本，系統會自動啟動多核心環境與 GPU 訓練：
```powershell
python train_ppo.py
```

### 監控模型收斂與成效
啟動 TensorBoard 查看即時的策略收斂情況（總報酬率與模型學習狀態）：
```powershell
tensorboard --logdir ./smc_tensorboard/
```

## 開發者
- **Author**: CJohn Lin
- **Version**: 1.0.0
