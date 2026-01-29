import ccxt
import pandas as pd

# 初始化交易所
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

try:
    # 尝试获取行情测试连接（不需要API Key也能测，如果这个通了说明网络通了）
    print("正在测试网络连接...")
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"当前 BTC 价格: {ticker['last']}")
    
    # 如果上面成功，再测试账户余额（需要正确的API Key）
    # print(exchange.fetch_balance())
    
except Exception as e:
    print(f"连接依然失败: {e}")