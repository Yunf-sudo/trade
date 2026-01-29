import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm import tqdm  # 进度条

# 1. 配置交易所
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

def download_history(symbol='BTC/USDT', timeframe='1h', days=365*2):
    """
    下载过去 N 天的数据
    """
    # 计算开始时间 (毫秒时间戳)
    start_date = datetime.now() - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    print(f"🚀 开始下载 {symbol} 过去 {days} 天的数据...")
    
    # 估算大概需要请求多少次
    total_intervals = (days * 24) / 100  # 假设每次取100条
    pbar = tqdm(total=int(total_intervals))

    while True:
        try:
            # 获取数据
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=100)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # 更新下一次获取的起始时间 (最后一根K线的时间 + 1毫秒)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            pbar.update(1)
            
            # 如果获取到了当前时间，就停止
            if last_timestamp >= int(datetime.now().timestamp() * 1000) - 3600000:
                break
                
            # 稍微休息一下，防止被交易所封IP
            # ccxt开启 rateLimit 后会自动处理，但为了保险加一点
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"⚠️ 下载中断: {e}")
            time.sleep(5) # 报错了就多睡一会再试
            continue

    pbar.close()
    
    # 转为 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 去重（防止网络波动导致的数据重复）
    df = df.drop_duplicates(subset=['timestamp'])
    
    filename = 'btc_history_2y.csv'
    df.to_csv(filename, index=False)
    print(f"\n✅ 数据下载完成！共 {len(df)} 条K线，已保存为 {filename}")
    return df

if __name__ == "__main__":
    download_history()