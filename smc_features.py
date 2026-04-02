import pandas as pd
import numpy as np

class SMCFeatures:
    """
    SMC (Smart Money Concept) 特徵提取類別。
    以完全向量化 (Vectorized) 的方式實作，提升在大資料量下的運算速度。
    """
    def __init__(self, df: pd.DataFrame):
        """
        初始化並複製 DataFrame，避免修改到原始數據。
        輸入 DataFrame 需要包含 'Open', 'High', 'Low', 'Close', 'Volume' 欄位。
        """
        # 確保不會修改到原始傳入的 DataFrame
        self.df = df.copy()
        
    def extract_fvg(self):
        """
        辨識三根 K 線構成的失衡區間 (Fair Value Gap)。
        - Bullish FVG: 第1根的高點 < 第3根的低點
        - Bearish FVG: 第1根的低點 > 第3根的高點
        """
        df = self.df
        
        # Bullish FVG: 當前 K 線 (第3根) 的 Low > 前2根 (第1根) 的 High
        bullish_fvg = df['Low'] > df['High'].shift(2)
        
        # Bearish FVG: 當前 K 線 (第3根) 的 High < 前2根 (第1根) 的 Low
        bearish_fvg = df['High'] < df['Low'].shift(2)
        
        # 初始化 fvg_active 欄位: 1 為 Bullish FVG, -1 為 Bearish FVG, 0 為無
        df['fvg_active'] = 0
        df.loc[bullish_fvg, 'fvg_active'] = 1
        df.loc[bearish_fvg, 'fvg_active'] = -1
        
        # 記錄 FVG 缺口區間的上下界
        df['fvg_upper'] = np.nan
        df['fvg_lower'] = np.nan
        
        # Bullish FVG 缺口區間：下界是第1根的高點，上界是第3根的低點
        df.loc[bullish_fvg, 'fvg_lower'] = df['High'].shift(2)
        df.loc[bullish_fvg, 'fvg_upper'] = df['Low']
        
        # Bearish FVG 缺口區間：下界是第3根的高點，上界是第1根的低點
        df.loc[bearish_fvg, 'fvg_lower'] = df['High']
        df.loc[bearish_fvg, 'fvg_upper'] = df['Low'].shift(2)
        
        self.df = df
        return self

    def extract_fractals(self):
        """
        辨識局部高低點 (Williams Fractals)。
        定義：中間 K 線的高(低)點高(低)於左右各兩根 K 線。
        
        注意：Fractal 是發生在當前 K 線的前 2 根，這在標記當下會產生未來函數 (Look-ahead bias)。
        為訓練 DRL 模型，實務上我們通常只在「確認後」才當作可用訊號使用。
        本函式忠實還原 Fractal 的發生點 (即中間 K 線位置)。
        """
        df = self.df
        
        # Fractal High: t 高於 t-1, t-2, t+1, t+2
        is_fractal_high = (
            (df['High'] > df['High'].shift(1)) & 
            (df['High'] > df['High'].shift(2)) & 
            (df['High'] > df['High'].shift(-1)) & 
            (df['High'] > df['High'].shift(-2))
        )
        
        # Fractal Low: t 低於 t-1, t-2, t+1, t+2
        is_fractal_low = (
            (df['Low'] < df['Low'].shift(1)) & 
            (df['Low'] < df['Low'].shift(2)) & 
            (df['Low'] < df['Low'].shift(-1)) & 
            (df['Low'] < df['Low'].shift(-2))
        )
        
        # 標記發生點 (True / False)
        df['fractal_high'] = is_fractal_high
        df['fractal_low'] = is_fractal_low
        
        # 記錄 Fractal 發生時的價格，供後續 MSS 使用
        df['fractal_high_price'] = np.where(is_fractal_high, df['High'], np.nan)
        df['fractal_low_price'] = np.where(is_fractal_low, df['Low'], np.nan)
        
        self.df = df
        return self
        
    def extract_mss(self):
        """
        辨識 Market Structure Shift (MSS)。
        定義：當價格突破前一個相反方向的 Fractal 時，標記為結構轉變。
        為符合 DRL/量化交易中無未來函數的嚴格要求，突破檢測基於歷史上「已確認」的 Fractal。
        """
        df = self.df
        
        if 'fractal_high_price' not in df.columns:
            self.extract_fractals()
            
        # 為了避免未來函數 (Look-ahead bias)：
        # t 時刻發生的 Fractal，必須等到 t+2 才能被確認 (因為需要右邊兩根 K 線)
        # 所以向後位移 2 格後進行 ffill，這代表「目前時間點我們已經『確認』的最新 Fractal 價格」
        recent_fractal_high = df['fractal_high_price'].shift(2).ffill()
        recent_fractal_low = df['fractal_low_price'].shift(2).ffill()
        
        # Bullish MSS: 當下收盤價突破最近已確認的 Fractal High
        # 使用 shift(1) 反向檢查前一根是不在突破狀態之下，來精準找出「突破發生的那一瞬間」
        bullish_break = df['Close'] > recent_fractal_high
        bullish_mss = bullish_break & (~bullish_break.shift(1).fillna(False))
        
        # Bearish MSS: 當下收盤價跌破最近已確認的 Fractal Low
        bearish_break = df['Close'] < recent_fractal_low
        bearish_mss = bearish_break & (~bearish_break.shift(1).fillna(False))
        
        # 初始化 is_mss 欄位: 1 為 Bullish MSS, -1 為 Bearish MSS, 0 為無
        df['is_mss'] = 0
        df.loc[bullish_mss, 'is_mss'] = 1
        df.loc[bearish_mss, 'is_mss'] = -1
        
        self.df = df
        return self

    def extract_all(self) -> pd.DataFrame:
        """
        一次性提取所有 SMC 特徵並回傳結果 DataFrame。
        """
        self.extract_fvg()
        self.extract_fractals()
        self.extract_mss()
        return self.df

# ===== 簡單測試範例 =====
if __name__ == "__main__":
    # 建立假數據測試
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100)
    data = {"Open": np.random.uniform(100, 200, 100)}
    df = pd.DataFrame(data, index=dates)
    df["High"] = df["Open"] + np.random.uniform(0, 10, 100)
    df["Low"] = df["Open"] - np.random.uniform(0, 10, 100)
    df["Close"] = df["Open"] + np.random.uniform(-5, 5, 100)
    df["Volume"] = np.random.randint(1000, 5000, 100)
    
    smc = SMCFeatures(df)
    features_df = smc.extract_all()
    
    print("FVG 特徵:")
    print(features_df[features_df['fvg_active'] != 0][['High', 'Low', 'fvg_active', 'fvg_upper', 'fvg_lower']].head())
    
    print("\nFractals 特徵:")
    print(features_df[(features_df['fractal_high']) | (features_df['fractal_low'])][['High', 'Low', 'fractal_high', 'fractal_low']].head())
    
    print("\nMSS 特徵:")
    print(features_df[features_df['is_mss'] != 0][['Close', 'is_mss']].head())
