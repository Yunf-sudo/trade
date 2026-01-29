import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
import matplotlib.pyplot as plt

# --- 1. 读取数据 (确保你已经下载了 4h 数据) ---
# 如果没有 csv，请先运行上一段代码的 download_4h_data()
df = pd.read_csv('btc_4h_data.csv') 

# --- 2. 特征工程 (升级版) ---
# A. 趋势
df['SMA_20'] = df['close'].rolling(20).mean()
df['SMA_50'] = df['close'].rolling(50).mean()
# 价格与均线的距离 (这是 AI 判断位置的核心)
df['Dist_SMA20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
df['Dist_SMA50'] = (df['close'] - df['SMA_50']) / df['SMA_50']

# B. 动量 (RSI)
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
df['RSI'] = get_rsi(df['close'])

# C. 波动率 (ATR - 衡量当前市场是否活跃)
# 如果 ATR 很低，说明是死鱼盘，不应该交易
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
df['ATR'] = true_range.rolling(14).mean()
# 相对 ATR (波动率占价格的比例)
df['ATR_Pct'] = df['ATR'] / df['close']

# D. 成交量力度
df['Vol_MA'] = df['volume'].rolling(20).mean()
df['Vol_Ratio'] = df['volume'] / df['Vol_MA'] # 当前量是均量的多少倍

# --- 3. 目标定义 (修正：降低门槛，移除人工过滤) ---
# 逻辑：未来 3根K线 (12小时) 内，最大涨幅 > 1.0%
# 注意：我们用 high 来计算最大涨幅，而不是 close，这更符合止盈逻辑
df['Future_High'] = df['high'].shift(-1).rolling(3).max()
df['Future_Return'] = df['Future_High'] / df['close'] - 1

# 目标：只要未来12小时内最高点涨幅超过 1%，就算赢 (Label=1)
df['Target'] = (df['Future_Return'] > 0.01).astype(int)

df.dropna(inplace=True)

# --- 4. 训练 XGBoost ---
features = ['Dist_SMA20', 'Dist_SMA50', 'RSI', 'ATR_Pct', 'Vol_Ratio']
X = df[features]
y = df['Target']

# 打印一下正样本比例，确保不是 0.001
print(f"正样本比例 (赚钱机会占比): {y.mean():.2%}")
if y.mean() < 0.1:
    print("⚠️ 警告：赚钱机会太少，模型很难训练！")

split = int(len(X) * 0.85)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 关键：计算 scale_pos_weight
# 如果正样本少，我们要增加它的权重
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_weight = neg_count / pos_count

print(f"训练 XGBoost (权重补偿: {scale_weight:.2f})...")

model = xgb.XGBClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=5, 
    eval_metric='logloss',
    scale_pos_weight=scale_weight,
    subsample=0.8,         # 每次只用80%数据，防止过拟合
    colsample_bytree=0.8   # 每次只用80%特征
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# --- 5. 评估 ---
probs = model.predict_proba(X_test)[:, 1]

print("\n" + "="*45)
print(f"{'XGBoost v3.0 (无人工过滤版)':^45}")
print("="*45)
print(f"{'阈值 (Prob)':<12} | {'准确率 (Precision)':<18} | {'信号数':<10} | {'参考胜率'}")
print("-" * 45)

best_prec = 0
for t in [0.5, 0.6, 0.65, 0.7, 0.75]:
    preds = (probs > t).astype(int)
    count = np.sum(preds)
    if count > 0:
        prec = precision_score(y_test, preds)
        print(f"{t:<12} | {prec:<18.2%} | {count:<10} | {prec:.2%}")
    else:
        print(f"{t:<12} | {'无信号':<18} | 0")

xgb.plot_importance(model, title='Feature Importance (v3.0)')
plt.show()