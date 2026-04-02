import numpy as np
import pandas as pd

# 若您有安裝 gymnasium，可正常執行以下匯入。若無，請使用 pip install gymnasium
import gymnasium as gym
from gymnasium import spaces

# 載入我們第一階段實作的特徵提取器
from smc_features import SMCFeatures

def compute_mtf_rsi(series, period=14):
    """
    簡單計算 RSI 特徵，避開外部依賴 (talib) 的麻煩。
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=period, min_periods=1).mean()
    
    # 避免除以零
    rs = np.where(loss == 0, 100.0, gain / loss)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index).fillna(50)

class SMCEnv(gym.Env):
    """
    針對 SMC (Smart Money Concept) 回測與 DRL 訓練設計的客製化環境。
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, df: pd.DataFrame, window_size=50, initial_balance=10000.0,
                 transaction_cost=0.001, beta_drawdown=0.5):
        super(SMCEnv, self).__init__()
        
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # beta_drawdown 決定 Agent 在遇到回撤時的懲罰力道
        self.beta_drawdown = beta_drawdown
        
        # Action space: [-1.0, 1.0] 連續空間
        # -1.0 代表 100% 全倉做空, 1.0 代表 100% 全倉做多
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State space: 維度為 [視窗長度, 特徵數量]
        # 使用 9 個特徵: Open, High, Low, Close, Volume, FVG_Width, is_mss, RSI_14, RSI_56
        self.num_features = 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.num_features), 
            dtype=np.float32
        )
        
        self._preprocess_data()
        self.reset()
        
    def _preprocess_data(self):
        """
        初始化環境時，將 DataFrame 加上 SMC / RSI 等特徵並轉換為 numpy array 以加速運算。
        """
        # 1. 提取第一階段建立的 SMC 特徵
        smc = SMCFeatures(self.df)
        self.df = smc.extract_all()
        
        # 2. 計算衍生特徵：FVG 區間寬度
        self.df['fvg_width'] = 0.0
        mask = self.df['fvg_active'] != 0
        self.df.loc[mask, 'fvg_width'] = (self.df.loc[mask, 'fvg_upper'] - self.df.loc[mask, 'fvg_lower']).abs()
        
        # 3. 多時框 RSI 模擬計算 (14 為標準, 56 為高時間層級 4x)
        self.df['rsi_14'] = compute_mtf_rsi(self.df['Close'], 14)
        self.df['rsi_56'] = compute_mtf_rsi(self.df['Close'], 56)
        
        # 4. 填補空值 (避免神經網路輸入 NaN 導致梯度爆炸)
        self.df.bfill(inplace=True)
        self.df.fillna(0, inplace=True)
        
        # 特徵轉換與記憶體緩存
        self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'fvg_width', 'is_mss', 'rsi_14', 'rsi_56']
        self.data_mat = self.df[self.feature_cols].values
        self.close_prices = self.df['Close'].values
        self.fvg_actives = self.df['fvg_active'].values
        self.fvg_uppers = self.df['fvg_upper'].values
        self.fvg_lowers = self.df['fvg_lower'].values
    
    def reset(self, seed=None, options=None):
        """
        將環境重置到初始狀態，開啟新的 Episode。
        """
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_position = 0.0
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        """
        產生基礎常態化 (Normalization) 後的 State。
        """
        # 複製出當前視窗，避免改到原始環境資料
        obs = self.data_mat[self.current_step - self.window_size : self.current_step].copy()
        
        # 以最新一根的收盤價作為歸一化錨點
        current_close = self.close_prices[self.current_step - 1]
        
        if current_close > 0:
            # 價格常態化 (O, H, L, C, FVG_Width)
            obs[:, 0:4] = obs[:, 0:4] / current_close
            obs[:, 5] = obs[:, 5] / current_close
        
        # 量能常態化 (取 Log 平滑)
        obs[:, 4] = np.log1p(obs[:, 4])
        
        # RSI 縮放至 [0, 1] 區間 (因為 RSI 在 0~100)
        obs[:, 7:9] = obs[:, 7:9] / 100.0
        
        return obs.astype(np.float32)
        
    def step(self, action):
        """
        與環境互動，執行 Agent 送出的 action (-1 ~ 1)。
        """
        # 確保動作嚴格遵守在 -1.0 到 1.0 之間
        action = np.clip(action[0], -1.0, 1.0)
        
        prev_net_worth = self.net_worth
        current_price = self.close_prices[self.current_step]
        prev_price = self.close_prices[self.current_step - 1]
        
        # --- 資金管理核心邏輯 ---
        
        # 1. 結算上一期的持倉收益
        price_change_pct = (current_price - prev_price) / prev_price
        daily_return = self.current_position * price_change_pct * self.net_worth
        
        # 2. 計算目前動作改變產生的交易成本 (只有當持倉變化時才會產生手續費與滑價)
        trade_size = abs(action - self.current_position)
        txn_cost_amount = trade_size * self.transaction_cost * self.net_worth
        
        # 3. 更新當前總淨值與新的部位大小
        self.net_worth = self.net_worth + daily_return - txn_cost_amount
        self.current_position = action
        
        # 4. 更新高點並計算 Max Drawdown
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0.0
        
        # --- 獎勵計算 (Reward Function) ---
        
        # 核心 Reward (淨值變化率 - beta * MDD 懲罰)
        step_reward = (self.net_worth - prev_net_worth) / prev_net_worth
        mdd_penalty = self.beta_drawdown * drawdown
        reward = float(step_reward - mdd_penalty)
        
        # 加入 SMC (Smart Money Concept) 的客製化 Shaping Reward
        # 目的：激勵模型學習「在 FVG 區間內產生順勢作法」的交易策略
        active_fvg = self.fvg_actives[self.current_step]
        
        if trade_size > 0.1 and active_fvg != 0:
            upper = self.fvg_uppers[self.current_step]
            lower = self.fvg_lowers[self.current_step]
            
            # 若目前價格落在 FVG 供需區間內
            if lower <= current_price <= upper:
                # Bullish FVG (支撐) + 建多倉 => 給予額外 +0.05 獎勵
                if active_fvg == 1 and action > self.current_position:
                    reward += 0.05
                # Bearish FVG (壓力) + 建空倉 => 給予額外 +0.05 獎勵
                elif active_fvg == -1 and action < self.current_position:
                    reward += 0.05
                    
        # 步進並判斷是否結束
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            "net_worth": self.net_worth,
            "drawdown": drawdown,
            "position": self.current_position,
            "step_reward": step_reward
        }
        
        return self._get_observation(), reward, done, truncated, info


# ============================================
# 平行化環境工廠函式 (適配 Intel 12700K 多核心)
# ============================================

def make_env(df, **kwargs):
    """
    返回可建立環境的 Factory 函數。
    供 SubprocVecEnv 初始化子進程環境使用。
    """
    def _init():
        return SMCEnv(df, **kwargs)
    return _init

def create_vectorized_envs(df, num_envs=8, **kwargs):
    """
    提供給 Intel 12700K 榨乾效能的多進程平行池。(利用 Stable-Baselines3 特性)
    建議 num_envs 可設為 8~16。
    """
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        # 將建立 n 個子進程，各自跑一段不同的 SMCEnv
        return SubprocVecEnv([make_env(df, **kwargs) for _ in range(num_envs)])
    except ImportError:
        raise ImportError("若要啟用多進程環境，請先執行: pip install stable-baselines3")


if __name__ == "__main__":
    # ===== 單體運行自我檢測 =====
    print("Initializing synthetic testing data...")
    dates = pd.date_range("2026-01-01", periods=1000)
    data = {"Open": np.random.uniform(100, 200, 1000)}
    df = pd.DataFrame(data, index=dates)
    df["High"] = df["Open"] + np.random.uniform(0, 10, 1000)
    df["Low"] = df["Open"] - np.random.uniform(0, 10, 1000)
    df["Close"] = df["Open"] + np.random.uniform(-5, 5, 1000)
    df["Volume"] = np.random.randint(1000, 5000, 1000)
    
    try:
        from gymnasium.utils.env_checker import check_env
        env = SMCEnv(df)
        print("Checking environment with Gymnasium's check_env API...")
        # check_env 如果沒有報錯代表環境符合所有標準要求
        check_env(env)
        print("✅ Environment conforms to Gymnasium API standards!")
        
        obs, info = env.reset()
        print(f"Observation Shape: {obs.shape}")
        
        print("\n--- Running 5 purely random steps ---")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1} | Action: {action[0]:+5.2f} | Reward: {reward:+6.4f} | Net Worth: {info['net_worth']:.2f}")
            if done:
                break
        print("✅ All local tests completed successfully.")
    except ImportError as e:
        print("\n⚠️", e)
        print("Please ensure you have gymnasium installed to run the tests: pip install gymnasium")
