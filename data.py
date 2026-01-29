import ccxt
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm

def get_fear_greed_history(limit=2000):
    """
    抓取“贪婪恐慌指数”历史数据 (代表新闻/舆论/政治影响的综合情绪)
    来源: alternative.me
    """
    print("正在抓取市场情绪(新闻/舆论)数据...")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    response = requests.get(url)
    data = response.json()['data']
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    # 只需要日期部分，方便和K线对齐
    df['date'] = df['timestamp'].dt.date
    df['fng_value'] = df['value'].astype(int)
    df['fng_classification'] = df['value_classification']
    return df[['date', 'fng_value']]

def download_market_data(symbol='BTC/USDT', days=365*4):
    """
    下载 OKX 日线数据 (1d)
    """
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
    
    print(f"正在下载 {symbol} 过去 {days} 天的日线数据...")
    start_date = datetime.now() - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=100)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000) - 86400000: break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date
    return df

if __name__ == "__main__":
    # 1. 获取价格
    df_price = download_market_data()
    
    # 2. 获取情绪
    df_fng = get_fear_greed_history(limit=len(df_price) + 100)
    
    # 3. 数据融合 (通过日期合并)
    # 这就是所谓的“多模态”数据的雏形
    df_merged = pd.merge(df_price, df_fng, on='date', how='left')
    
    # 填补空缺值 (以前没有指数的时候用50中性填充)
    df_merged['fng_value'] = df_merged['fng_value'].fillna(50)
    
    # 保存
    df_merged.to_csv('btc_sentiment_data.csv', index=False)
    print(f"✅ 数据融合完成！包含价格与市场情绪变量，已保存为 btc_sentiment_data.csv")
    print(df_merged.tail())