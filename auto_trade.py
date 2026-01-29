import ccxt
import pandas as pd
import joblib
import requests
import time
from datetime import datetime

# 全局配置
API_KEY = 'eff43a6a-84df-43e6-ad81-2fa9f2797d74'
SECRET = 'D4AE6BC2122A31EF96CAFBAD3F03FF9F'
PASSWORD = '@Aqjnr998'
SYMBOL = 'BTC/USDT'
AMOUNT = 0.0001 # 每次买入的数量

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

def get_realtime_features():
    """获取当前的实时特征数据"""
    # 1. 获取最近 60天 的日线 (为了计算均线)
    ohlcv = exchange.fetch_ohlcv(SYMBOL, '1d', limit=60)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 2. 获取最新情绪指数
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=15")
        fng_data = resp.json()['data']
        current_fng = int(fng_data[0]['value']) # 今天的
        # 这里需要构建历史序列来计算 fng_ma10
        fng_vals = [int(x['value']) for x in fng_data]
        fng_series = pd.Series(fng_vals[::-1]) #以此推算指标
    except:
        print("情绪接口获取失败，使用默认值")
        current_fng = 50
        fng_series = pd.Series([50]*15)

    # 3. 实时计算特征 (必须和训练时一模一样)
    # 我们只关心最后一行的数据
    current_close = df['close'].iloc[-1]
    sma_50 = df['close'].rolling(50).mean().iloc[-1]
    
    features = {
        'fng_value': current_fng,
        'fng_change': fng_series.diff().iloc[-1],
        'fng_ma10': fng_series.rolling(10).mean().iloc[-1],
        'dist_sma50': (current_close - sma_50) / sma_50,
        'volatility': df['close'].pct_change().rolling(30).std().iloc[-1],
        'vol_change': (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]
    }
    
    # 转为 DataFrame (单行)
    # 注意列的顺序必须和训练时 Features 列表完全一致
    feature_cols = ['fng_value', 'fng_change', 'fng_ma10', 'dist_sma50', 'volatility', 'vol_change']
    input_df = pd.DataFrame([features], columns=feature_cols)
    
    return input_df, current_close

def run_bot():
    print(f"🤖 10天长周期 AI 交易机器人启动 - {datetime.now()}")
    
    # 加载模型
    try:
        model = joblib.load('crypto_model_10d.pkl')
    except:
        print("❌ 找不到模型文件，请先运行 2_train_model.py")
        return

    while True:
        try:
            print("\n正在分析市场...")
            X_live, current_price = get_realtime_features()
            
            # 预测
            # predict_proba 返回 [[跌概率, 涨概率]]
            prob_up = model.predict_proba(X_live)[0][1]
            
            print(f"当前价格: {current_price} | 贪婪指数: {X_live['fng_value'].values[0]}")
            print(f"AI 预测未来10天上涨概率: {prob_up:.2%}")
            
            # 交易逻辑
            # 设置一个高门槛，比如 70% 把握才出手
            if prob_up > 0.70:
                print("🚀 信号触发！AI 极度看好，执行买入...")
                # exchange.create_market_buy_order(SYMBOL, AMOUNT) # 实盘请取消注释
            else:
                print("💤 信号微弱，继续空仓观望...")
            
            # 既然是日线策略，不需要每秒运行，每天运行一次即可
            # 为了演示，这里设置长睡眠
            print("等待下一次检查...")
            time.sleep(3600 * 4) # 每4小时检查一次即可
            
        except Exception as e:
            print(f"运行出错: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot()