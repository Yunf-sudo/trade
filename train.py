import pandas as pd
import numpy as np
import xgboost as xgb
import joblib  # 用于保存模型
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('btc_sentiment_data.csv')
    
    # --- 特征工程 (加入政治/情绪维度) ---
    
    # 1. 情绪特征
    # 情绪的变化率：是不是突然变得极其贪婪？
    df['fng_change'] = df['fng_value'].diff()
    # 情绪均线：长期情绪趋势
    df['fng_ma10'] = df['fng_value'].rolling(10).mean()
    
    # 2. 技术特征 (大周期)
    df['pct_change'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['sma_50'] = df['close'].rolling(50).mean()
    # 距离牛熊线的距离
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # 3. 波动率 (大事件通常伴随高波动)
    df['volatility'] = df['pct_change'].rolling(30).std()
    
    # --- 核心：定义 10天 周期目标 ---
    # 未来 10 天后的收盘价
    prediction_days = 10
    df['future_close'] = df['close'].shift(-prediction_days)
    
    # 目标：10天后涨幅超过 5% (0.05) 记为 1 (买入机会)
    # 既然考虑政治因素，我们只抓大行情，不抓小波动
    df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.05).astype(int)
    
    df.dropna(inplace=True)
    
    # 选取特征变量
    features = ['fng_value', 'fng_change', 'fng_ma10', 'dist_sma50', 'volatility', 'vol_change']
    X = df[features]
    y = df['target']
    
    # 划分训练集 (不打乱时间顺序)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 计算正负样本权重 (防止数据不平衡)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    print(f"开始训练... 训练集数量: {len(X_train)} | 测试集数量: {len(X_test)}")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 保存模型和特征列表，供后续交易使用
    joblib.dump(model, 'crypto_model_10d.pkl')
    print("✅ 模型训练完成，已保存为 crypto_model_10d.pkl")
    
    # 顺便输出一下特征重要性
    import matplotlib.pyplot as plt
    xgb.plot_importance(model, title='Feature Importance (Including Sentiment)')
    plt.show()

if __name__ == "__main__":
    train()