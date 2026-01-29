import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import precision_score, classification_report
import matplotlib.pyplot as plt

def evaluate():
    # 1. 重新加载数据 (为了还原测试集)
    # 注意：这里逻辑必须和训练时完全一致
    df = pd.read_csv('btc_sentiment_data.csv')
    
    # 特征工程复现
    df['fng_change'] = df['fng_value'].diff()
    df['fng_ma10'] = df['fng_value'].rolling(10).mean()
    df['pct_change'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['volatility'] = df['pct_change'].rolling(30).std()
    
    prediction_days = 10
    df['future_close'] = df['close'].shift(-prediction_days)
    df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.05).astype(int)
    df.dropna(inplace=True)
    
    features = ['fng_value', 'fng_change', 'fng_ma10', 'dist_sma50', 'volatility', 'vol_change']
    X = df[features]
    y = df['target']
    
    # 取最后 20% 作为测试集
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    
    # 2. 加载模型
    model = joblib.load('crypto_model_10d.pkl')
    
    # 3. 预测概率
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n========= 10天周期策略评估报告 =========")
    
    # 寻找最佳阈值
    for t in [0.5, 0.6, 0.7, 0.8]:
        preds = (probs > t).astype(int)
        count = np.sum(preds)
        if count > 0:
            prec = precision_score(y_test, preds)
            print(f"阈值 > {t} | 信号次数: {count} | 10天后上涨概率(查准率): {prec:.2%}")
        else:
            print(f"阈值 > {t} | 无信号")

    # 可视化概率分布
    plt.hist(probs, bins=50)
    plt.title('Prediction Probabilities Distribution')
    plt.show()

if __name__ == "__main__":
    evaluate()