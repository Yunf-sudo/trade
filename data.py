import ccxt
import pandas as pd
import time
from datetime import datetime

# 1. 初始化交易所 (使用你刚才测试成功的配置)
exchange = ccxt.okx({
    'apiKey': 'eff43a6a-84df-43e6-ad81-2fa9f2797d74',
    'secret': 'D4AE6BC2122A31EF96CAFBAD3F03FF9F',
    'password': '@Aqjnr998',
    'enableRateLimit': True, # 防止请求过快被封IP
# --- 核心修改部分 ---
    'proxies': {
        'http': 'http://127.0.0.1:7897',  # 注意：这里换成你的代理端口
        'https': 'http://127.0.0.1:7897', # 注意：https请求也走http代理协议
    },
})

def fetch_ohlcv_data(symbol='BTC/USDT', timeframe='1h', limit=500):
    print(f"正在下载 {symbol} 的 {timeframe} K线数据...")
    
    # 获取K线数据: [时间戳, 开盘, 最高, 最低, 收盘, 成交量]
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # 转换为表格格式
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 处理时间戳 (把毫秒转换为人类能看懂的时间)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

# 2. 执行下载
try:
    # 下载最近 500 根 1小时级别的 K线
    df = fetch_ohlcv_data('BTC/USDT', '1h', 500)
    
    # 3. 简单计算一个技术指标 (例如：移动平均线)
    # AI 通常需要这些指标作为输入特征
    df['SMA_20'] = df['close'].rolling(window=20).mean() # 20周期均线
    
    print("\n下载成功！数据预览：")
    print(df.tail(5)) # 打印最后5行看看
    
    # 4. 保存到文件
    filename = 'btc_1h_data.csv'
    df.to_csv(filename, index=False)
    print(f"\n数据已保存到桌面或当前目录下的: {filename}")
    
except Exception as e:
    print(f"发生错误: {e}")