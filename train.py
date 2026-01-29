import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- 1. 加载数据 ---
print("正在加载数据...")
df = pd.read_csv('btc_history_2y.csv')

# --- 2. 特征工程 (关键修改：使用收益率而非绝对价格) ---
# 计算对数收益率 (Log Return)，这是金融建模的标准
# 它能把非平稳的价格序列变成平稳序列
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# 波动率特征
df['volatility'] = df['log_ret'].rolling(window=20).std()

# 动量特征 (RSI)
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = get_rsi(df['close'])

# 均线偏离度 (价格距离均线有多远)
df['sma_dist'] = (df['close'] - df['close'].rolling(50).mean()) / df['close']

# 清洗空值
df.dropna(inplace=True)

# --- 3. 重新定义目标 (Target) ---
# 只有当下一小时涨幅 > 0.25% (0.0025) 时，才标记为 1 (买入机会)
# 这样 AI 就不会被迫去预测那些无意义的震荡
threshold = 0.0025 
df['future_ret'] = df['close'].shift(-1) / df['close'] - 1
df['Target'] = (df['future_ret'] > threshold).astype(int)

# 检查一下正负样本比例
print(f"样本分布: {Counter(df['Target'])}")
# 如果 1 太少 (比如只有 10%)，模型会很难训练。理想情况是 1 占比 30%-40%。

# --- 4. 准备输入数据 ---
# 我们选取这几个“平稳”的特征
features = ['log_ret', 'volatility', 'rsi', 'sma_dist']
data = df[features].values
target = df['Target'].values

# 标准化 (StandardScaler 比 MinMax 更适合由于正态分布的数据)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 构建时间窗
X = []
y = []
lookback = 48 # 缩短一点，看过去48小时

for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i-lookback:i])
    y.append(target[i])

X, y = np.array(X), np.array(y)

# 划分数据集 (这次我们不乱序，保留时间顺序)
split = int(len(X) * 0.85) # 85% 训练
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 计算类别权重 (如果暴涨的机会很少，我们要告诉 AI 那个 1 很珍贵)
# 这能防止 AI 偷懒全猜 0
total = len(y_train)
pos = np.sum(y_train)
neg = total - pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

# --- 5. 改进的模型结构 ---
model = Sequential()
# 第一层：更多神经元，加 L2 正则化或是 BatchNormalization
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3)) 

# 第二层
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 使用更小的学习率
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# --- 6. 训练 ---
print("🚀 开始训练 v2.0 (带阈值过滤)...")
# EarlyStopping: 如果训练不准了，自动提前停止
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    class_weight=class_weight, # 这一步很关键，解决样本不平衡
    callbacks=[early_stop],
    verbose=1
)

# --- 7. 评估 ---
print("\n" + "="*30)
res = model.evaluate(X_test, y_test)
print(f"准确率 (Accuracy): {res[1]:.2%}")
print(f"查准率 (Precision - AI说涨真的涨的概率): {res[2]:.2%}")
print("="*30)

# 模拟信号分布
preds = model.predict(X_test)
print(f"测试集预测信号分布: 超过0.5的比例: {np.mean(preds > 0.5):.2%}")

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# 1. 获取模型对测试集的预测概率 (0到1之间的小数)
# 注意：这一步不需要重新训练，直接用刚才训练好的 model
pred_probs = model.predict(X_test, verbose=0)

print(f"\n{'阈值 (Threshold)':<15} | {'查准率 (Precision)':<18} | {'交易次数 (Signals)':<15} | {'胜率 (Win Rate)':<15}")
print("-" * 70)

best_threshold = 0.5
best_precision = 0.0

# 2. 循环测试不同的门槛
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    # 如果概率 > t，则标记为 1 (买入)，否则为 0
    my_preds = (pred_probs > t).astype(int)
    
    # 只要有过至少一次买入信号，才计算
    if np.sum(my_preds) > 0:
        prec = precision_score(y_test, my_preds, zero_division=0)
        count = np.sum(my_preds)
        
        # 简单计算一下这部分信号里的实际胜率 (和 Precision 类似)
        print(f"{t:<18.2f} | {prec:<22.2%} | {count:<19} | {prec:.2%}")
        
        if prec > best_precision and count > 10: # 至少要有10次交易才有统计意义
            best_precision = prec
            best_threshold = t
    else:
        print(f"{t:<18.2f} | {'无交易信号':<22} | 0")

print("-" * 70)
print(f"🚀 最佳策略建议：将买入阈值设定为 > {best_threshold}")
print(f"预期胜率可提升至: {best_precision:.2%}")

# --- 可视化概率分布 ---
# 看看 AI 到底有多少次是非常确定的？
plt.figure(figsize=(10, 5))
plt.hist(pred_probs, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.title('AI Prediction Probability Distribution')
plt.xlabel('Probability (0=Bearish, 1=Bullish)')
plt.ylabel('Count')
plt.axvline(x=best_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Best Threshold {best_threshold}')
plt.legend()
plt.show()