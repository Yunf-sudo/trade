import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. 读取你刚才下载的数据
df = pd.read_csv('btc_1h_data.csv')

# --- 2. 特征工程 (Feature Engineering) ---
# 我们需要手动计算一些指标喂给 AI，而不仅仅是价格

# 简单移动平均线 (SMA)
df['SMA_10'] = df['close'].rolling(window=10).mean()
df['SMA_20'] = df['close'].rolling(window=20).mean()

# 相对强弱指标 (RSI) - 手动计算简化版
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['close'])

# 收益率 (Return)
df['Return'] = df['close'].pct_change()

# 波动率 (Volatility)
df['Volatility'] = df['Return'].rolling(window=20).std()

# --- 3. 构建目标 (Labeling) ---
# 核心问题：下一小时收盘价是否比当前高？
# shift(-1) 读取的是“下一行”的数据
df['Next_Close'] = df['close'].shift(-1)
df['Target'] = (df['Next_Close'] > df['close']).astype(int) 
# 1 代表涨，0 代表跌/平

# --- 4. 清洗数据 ---
# 因为计算指标会有 NaN (空值)，必须去掉，否则模型报错
df.dropna(inplace=True)

# 定义 AI 的输入 (X) 和 想要预测的答案 (y)
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'SMA_20', 'RSI', 'Return', 'Volatility']
X = df[feature_cols]
y = df['Target']

# --- 5. 划分训练集和测试集 ---
# 重要！！时间序列数据绝对不能打乱顺序 (shuffle=False)
# 我们用过去 80% 的数据训练，预测最近 20% 的未来
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"训练数据量: {len(X_train)} 行, 测试数据量: {len(X_test)} 行")

# --- 6. 训练模型 (The Training) ---
print("正在训练 AI 大脑 (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# --- 7. 验证结果 ---
# 让 AI 预测测试集
predictions = model.predict(X_test)

# 计算准确率
acc = accuracy_score(y_test, predictions)
print("-" * 30)
print(f"AI 预测准确率: {acc:.2%}")
print("-" * 30)

# 查看 AI 觉得哪个指标最重要
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nAI 认为最重要的特征因子：")
print(importances.head(5))

# 简单回测逻辑：如果 AI 预测涨(1) 我们就持有，否则空仓
df_test = df.iloc[split:].copy()
df_test['Prediction'] = predictions
# 策略收益 = 实际收益率 * 预测操作 (1或0)
df_test['Strategy_Return'] = df_test['Return'].shift(-1) * df_test['Prediction']

cum_market_return = (1 + df_test['Return'].shift(-1)).cumprod()
cum_strategy_return = (1 + df_test['Strategy_Return']).cumprod()

print(f"\n基准市场回报 (HODL): {cum_market_return.iloc[-2]:.4f}")
print(f"AI 策略回报: {cum_strategy_return.iloc[-2]:.4f}")