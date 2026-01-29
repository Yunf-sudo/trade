import numpy as np
import pandas as pd
import tensorflow as pd_tf # 别名处理
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- 1. 加载并增强数据 ---
print("正在处理数据...")
df = pd.read_csv('btc_history_2y.csv')

# 特征工程：添加技术指标
# AI 需要看到趋势，不仅仅是价格
df['SMA_15'] = df['close'].rolling(window=15).mean()
df['SMA_60'] = df['close'].rolling(window=60).mean()
df['Vol_Change'] = df['volume'].pct_change()

# RSI 计算
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True) # 去除计算产生的空值

# --- 2. 定义目标 ---
# 目标：预测下一个小时收盘价是涨(1) 还是 跌(0)
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

# 选取 AI 的输入特征
features = ['close', 'volume', 'SMA_15', 'SMA_60', 'RSI', 'Vol_Change']
data = df[features].values
target = df['Target'].values

# --- 3. 数据归一化 (非常重要) ---
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# --- 4. 构建时间序列数据 (Sliding Window) ---
# LSTM 需要看到历史片段。我们设定 lookback=60
# 意思是用 过去60小时的数据 -> 预测 第61小时的涨跌
X = []
y = []
lookback = 60

for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i-lookback:i]) # 过去60行所有特征
    y.append(target[i]) # 第i行的目标

X, y = np.array(X), np.array(y)

# 划分训练集和测试集 (前80%训练，后20%验证)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"构建完成：训练样本 {X_train.shape[0]}, 测试样本 {X_test.shape[0]}")

# --- 5. 搭建 LSTM 模型 ---
model = Sequential()

# 第一层 LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # 丢弃20%神经元防止过拟合

# 第二层 LSTM
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# 输出层 (Sigmoid 激活函数用于输出 0-1 之间的概率)
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 6. 开始训练 ---
print("🚀 开始训练神经网络 (这可能需要几分钟)...")
# epochs=20 (学20遍), batch_size=32 (每次学32个样本)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# --- 7. 评估结果 ---
print("\n" + "="*30)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"最终测试集准确率: {accuracy:.2%}")
print("="*30)

# --- 8. 简单的实战模拟 ---
# 获取模型预测的概率
predictions = model.predict(X_test)
# 如果概率 > 0.5 判为涨，否则判为跌
pred_labels = (predictions > 0.5).astype(int).flatten()

# 只是为了看最后几条的预测情况
result_df = pd.DataFrame({'Actual': y_test[-10:], 'Predicted': pred_labels[-10:], 'Prob': predictions[-10:].flatten()})
print("\n最后 10 个小时的预测对比 (Actual:1涨0跌):")
print(result_df)