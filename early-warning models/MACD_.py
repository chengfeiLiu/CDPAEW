import pandas as pd
import pandas_datareader as web
from datetime import datetime, timedelta
import talib
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.ticker as ticker
import yfinance
yfinance.pdr_override()
def stock(stock_code): 

    stock_info = pd.read_csv(filepath_or_buffer='./600036.SS_huicai.csv',encoding='utf-8')
    stock_info = stock_info.reset_index()  
    stock_info = stock_info.astype({"Date": str})    
    return stock_info
def get_indicators(stock_code):
    data = stock(stock_code)
    
    #获取macd
    data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data['Close'],fastperiod=8, slowperiod=24, signalperiod=7)
    data_result_write = pd.ExcelWriter("./macd_result_huicai.xlsx")
    data.to_excel(data_result_write)
    data_result_write.close()

    #获取10日均线和30日均线
    data["ma10"] = talib.MA(data["Close"], timeperiod=10)
    data["ma30"] = talib.MA(data["Close"], timeperiod=30)
 
    #获取rsi
    data["rsi"] = talib.RSI(data["Close"])
    return data

def industry(dict):
    for key, value in dict.items():
        # d.iteritems: an iterator over the (key, value) items
        stock_info = get_indicators(key)
        
if __name__ == '__main__':
    finance_list = {
    "01.ss": "fenchen"}
    industry(finance_list)