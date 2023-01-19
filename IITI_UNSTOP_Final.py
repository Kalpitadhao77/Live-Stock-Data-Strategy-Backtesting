#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import numpy as np
import yfinance as yf
import ipywidgets as widgets
from autocalc.autocalc import Var
import math
import streamlit as st
# Load data
showWarningOnDirectExecution = False
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Trading Strategy")
st.text("Welcome!")
activities=["US Stocks","Indian Stocks"]
choice=st.sidebar.selectbox("Select Country Stocks",activities)
if choice=="US Stocks":
    activities=["MNST","AAPL","MSFT","SPY","AMZN","GOOGL","JNJ","GOOG","UNH","XOM","JPM","NVDA","BRK.B","PG","V","HD","CVX","MA","LLY","ABBV","PFE","MRK","META","PEP","KO","BAC"]
if choice=="Indian Stocks":
    activities=["TCS.NS","ITC.NS","TATAMOTORS.NS","HINDALCO.NS","ONGC.NS","TATASTEEL.NS","COALINDIA.NS","JSWSTEEL.NS","WIPRO.NS","BPCL.NS","YESBANK.NS","IDEA.NS","PNB.NS","IOC.NS","ZOMATO.NS","UCOBANK.NS","IRFC.NS","HCC.NS","IDBI.NS"]
choice=st.sidebar.selectbox("Select Stock",activities)
st.write("1m gives data worth of 7 days")
st.write("2m,5m,15m,30m,90m gives data worth of 60 days")
st.write("1h gives data worth of 730 days")
intervals=["1d","1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "5d", "1wk", "1mo", "3mo"]
interval=st.selectbox("Select time interval for stock data",intervals)
period=["1000d","1d", "2d","5d", "7d", "15d", "30d", "60d", "90d", "200d", "500d",  "2000d", "5000d", "7000d","10000d"]
periods=st.selectbox("Select time period for stock data",period)
# period=st.slider("historical_data_period", min_value=1, max_value=7000, value=5000)
data = yf.download(tickers=choice, period=periods, interval=interval)
#Print data
data


# In[110]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
plt.plot(data.Close)
st.pyplot(fig=None, clear_figure=None)


# In[111]:


def ma(df, n):
    return pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))

def ema(df, n):
    return pd.Series(df['Close'].ewm(span=n,min_periods=n).mean(), name='EMA_' + str(n))

def mom(df, n):     
    return pd.Series(df.diff(n), name='Momentum_' + str(n))  

def roc(df, n):  
    M = df.diff(n - 1) ; N = df.shift(n - 1)  
    return pd.Series(((M / N) * 100), name = 'ROC_' + str(n)) 

def rsi(df, period):
    delta = df.diff().dropna()
    u = delta * 0; d = u.copy()
    u[delta > 0] = delta[delta > 0]; d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def sto(close, low, high, n,id): 
    stok = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    if(id is 0):
        return stok
    else:
        return stok.rolling(3).mean()


# In[112]:


def std_dev(data):
    # Get number of observations
    n = len(data)
    # Calculate mean
    mean = sum(data) / n
    # Calculate deviations from the mean
    deviations = sum([(x - mean)**2 for x in data])
    # Calculate Variance & Standard Deviation
    variance = deviations / (n - 1)
    s = variance**(1/2)
    return s


# In[113]:


def sharpe_ratio(data, risk_free_rate=0.0):
    # Calculate Average Daily Return
    mean_daily_return = sum(data) / len(data)
    # Calculate Standard Deviation
    s = std_dev(data)
    # Calculate Daily Sharpe Ratio
    daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
    # Annualize Daily Sharpe Ratio
    sharpe_ratio = 252**(1/2) * daily_sharpe_ratio
    
    return sharpe_ratio


# In[114]:


def maxprofit(sl,bl,q):
    maxp=0
    for i in range(len(sl)):
        maxp=max(maxp,(sl[i]-bl[i])*q[i])
    return maxp
def maxloss(sl,bl,q):
    minp=1e7
    for i in range(len(sl)):
        minp=min(minp,(sl[i]-bl[i])*q[i])
    return minp
def totalloss(sl,bl,q):
    loss=0
    for i in range(len(sl)):
        if sl[i]-bl[i]<0:
            loss+=(sl[i]-bl[i])*q[i]
    return loss
def totalprofit(sl,bl,q):
    p=0
    for i in range(len(sl)):
        if sl[i]-bl[i]>0:
            p+=(sl[i]-bl[i])*q[i]
    return p
def avgloss(sl,bl,q):
    loss=0
    c=1
    for i in range(len(sl)):
        if sl[i]-bl[i]<0:
            loss+=(sl[i]-bl[i])*q[i]
            c=c+1
    return loss/c
def avglossper(sl,bl,q):
    loss=0
    c=1
    for i in range(len(sl)):
        if sl[i]-bl[i]<0:
            loss+=(sl[i]-bl[i])*q[i]
            c=c+1
    if len(sl)>0:
        return (c/len(sl))*100
    return 0
def maxcontloss(sl,bl):
    maxlc=0
    ans=0
    for i in range(len(sl)):
        if sl[i]-bl[i]<0:
            maxlc=maxlc+1
        if sl[i]-bl[i]>=0:
            ans=max(ans,maxlc)
            maxlc=0
    return ans
def maxcontprofit(sl,bl):
    maxlc=0
    ans=0
    for i in range(len(sl)):
        if sl[i]-bl[i]>0:
            maxlc=maxlc+1
        if sl[i]-bl[i]<=0:
            ans=max(ans,maxlc)
            maxlc=0
    return ans
def largestprofit(sl,bl,q):
    maxlc=0
    for i in range(len(sl)):
        if sl[i]-bl[i]>0:
            maxlc=max(maxlc,(sl[i]-bl[i])*q[i])
    return maxlc
def largestloss(sl,bl,q):
    maxlc=0
    for i in range(len(sl)):
        if sl[i]-bl[i]<0:
            maxlc=min(maxlc,(sl[i]-bl[i])*q[i])
    return maxlc
def maxdrawdown(data):
    Roll_Max = data['Close'].rolling(20, min_periods=1).max()
    Daily_Drawdown = data['Close']/Roll_Max - 1.0

    Max_Daily_Drawdown = Daily_Drawdown.rolling(20, min_periods=1).min()
    return Max_Daily_Drawdown.max()
def max_drawdown_system(data):
    mdd = 0
    peak = data[0]
    for x in data:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd 
def trades(sl,bl,data):
    data['Bought at']=bl
    data['Sold at']=sl
    data[['Bought at','Sold at']]


# In[115]:


def macrossover(data,fast_period,slow_period,init_cash,transaction_cost,stop_loss):
    flag=0
    buy=0
    pp=0
    bl=[]
    sl=[]
    q=[]
    trades=0
    data['fast'] = ma(data,fast_period)
    data['slow'] = ma(data, slow_period)
    # shares=portfolio_size/data['Close'].mean()
    for i in range(len(data)):
        if data['fast'][i]>data['slow'][i] and flag==0 :
#             print("Buy at ")
#             print(data['Close'][i])
            buy=data['Close'][i]*(1+float(transaction_cost/100))
            q.append(init_cash/buy)
            bl.append(data['Close'][i])
            trades=trades+1
            flag=1
        if data['fast'][i]<data['slow'][i] and flag==1 :
#             print("sell at ")
#             print(data['Close'][i])
            sl.append(data['Close'][i])
            trades=trades+1
            flag=0
    fp=0
    avp=0
    postr=1
    for i in range(len(bl)-1):
        fp=fp+(sl[i]-bl[i])*q[i]
        if sl[i]-bl[i]>0:
            avp=avp+(sl[i]-bl[i])*q[i]
            postr=postr+1
        #print(fp)
    avp=avp/postr
    maslow=data.Close.rolling(slow_period).mean()
    mafast=data.Close.rolling(fast_period).mean()
    plt.figure(figsize=(20,12))
    plt.plot(data.Close)
    plt.plot(maslow,'r')
    plt.plot(mafast,'g')
    st.pyplot(fig=None, clear_figure=None)
    df=data
    df['Signal'] = 0.0
    df['Signal'] = np.where(df['fast'] > df['slow'], 1.0, 0.0)
    df['Position'] = df['Signal'].diff()
    plt.figure(figsize = (20,10))
    df['Close'].plot(color = 'k', label= 'Close Price') 
    df['fast'].plot(color = 'b',label = 'fast SMA') 
    df['slow'].plot(color = 'g', label = 'slow SMA')

    plt.plot(df[df['Position'] == 1].index, 
             df['fast'][df['Position'] == 1], 
             '^', markersize = 15, color = 'g', label = 'buy')

    plt.plot(df[df['Position'] == -1].index, 
             df['fast'][df['Position'] == -1], 
             'v', markersize = 15, color = 'r', label = 'sell')
    plt.ylabel('Price in Rupees', fontsize = 15 )
    plt.xlabel('Date', fontsize = 15 )
    plt.title('MAC', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()
    st.pyplot(fig=None, clear_figure=None)
#     plt.plot(ma25,'y')
    sharpe=sharpe_ratio(data['Close'])
    st.write("Total Transaction ",trades)
    st.write("Profit ",fp)
    st.write("Ending Capital ",fp+init_cash)
    st.write("Profit Percent",(fp/init_cash)*100)
    st.write("Total Transaction Costs ",trades*(transaction_cost/100))
    st.write("Sharpe Ratio",sharpe)
    st.write("Average Profit",avp)
    if len(sl)>0:
        st.write("Average profit percent",(postr/len(sl))*100)
    st.write("Max Profit",maxprofit(sl,bl,q))
    st.write("Max Loss",maxloss(sl,bl,q))
    st.write("Total Loss",totalloss(sl,bl,q))
    st.write("Total Profit",totalprofit(sl,bl,q))
    st.write("Average Loss",avgloss(sl,bl,q))
    st.write("Average Loss Percent",avglossper(sl,bl,q))
    st.write("Max Consecutive Loss",maxcontloss(sl,bl))
    st.write("Max Consecutive Profit",maxcontprofit(sl,bl))
#     st.write("Max Consecutive Loss",maxcontloss(sl,bl))
    st.write("Largest Profit",largestprofit(sl,bl,q))
    st.write("Largest Loss",largestloss(sl,bl,q))
    st.write("Max Trade Drawdown",maxdrawdown(data))
    st.write("Max System Drawdown",max_drawdown_system(data['Close']))
    st.write("Recovery Factor",abs(totalprofit(sl,bl,q)/max_drawdown_system(data['Close'])))
    if totalloss(sl,bl,q)>0:
        st.write("Profit Factor",abs(totalprofit(sl,bl,q)/totalloss(sl,bl,q)))
    if avgloss(sl,bl,q)>0:
        st.write("Payoff Ratio",abs(avp/avgloss(sl,bl,q)))
    #st.write("Here are the trades which happened on the historical data with your strategy and parametrs",trades(sl,bl,data))
    df = pd.DataFrame()
    df['Bought at']=bl
    df=df[:len(sl)]
    df['Sold at']=sl
    st.write("Here are the trades which happened on the historical data with your strategy and parametrs")
    df
    

def bb(data,bb_period,rsi_period,rsis,rsib,init_cash):
    data['TP'] = (data['Close'] + data['Low'] + data['High'])/3
    data['std'] = data['TP'].rolling(bb_period).std(ddof=0)
    data['MA-TP'] = data['TP'].rolling(bb_period).mean()
    data['BOLU'] = data['MA-TP'] + 2*data['std']
    data['BOLD'] = data['MA-TP'] - 2*data['std']
    data['RSI_Period'] = rsi(data['Close'], rsi_period)
    flag=0
    buy=0
    pp=0
    bl=[]
    sl=[]
    q=[]
    trades=0
#     init_cash=100000
    # shares=portfolio_size/data['Close'].mean()
    for i in range(len(data)):
        if data['BOLD'][i]>data['Close'][i] and flag==0 and data['RSI_Period'][i]<rsis:
#             print("Buy at ")
#             print(data['Close'][i])
            buy=data['Close'][i]
            q.append(init_cash/buy)
            bl.append(data['Close'][i])
            trades=trades+1
            flag=1
        if data['BOLU'][i]<data['Close'][i] and flag==1 and data['RSI_Period'][i]>rsib:
#             print("sell at ")
#             print(data['Close'][i])
            sl.append(data['Close'][i])
            trades=trades+1
            flag=0
    fp=0
    avp=0
    postr=1
    for i in range(len(bl)-1):
        fp=fp+(sl[i]-bl[i])*q[i]
        if sl[i]-bl[i]>0:
            avp=avp+(sl[i]-bl[i])*q[i]
            postr=postr+1
    avp=avp/postr
    data['Close'].plot(label = 'CLOSE PRICES', color = 'skyblue')
    data['BOLU'].plot(label = 'UPPER BB ', linestyle = '--', linewidth = 1, color = 'black')
    data['MA-TP'].plot(label = 'MIDDLE BB ', linestyle = '--', linewidth = 1.2, color = 'grey')
    data['BOLD'].plot(label = 'LOWER BB ', linestyle = '--', linewidth = 1, color = 'black')
    plt.legend(loc = 'upper left')
    plt.title('BOLLINGER BANDS')
    plt.show()
    st.pyplot(fig=None, clear_figure=None)
    sharpe=sharpe_ratio(data['Close'])
    st.write("Profit ",fp)
    st.write("Profit Percent",(fp/init_cash)*100)
    st.write("Sharpe Ratio",sharpe)
    st.write("Average Profit",avp)
    if len(sl)>0:
        st.write("Average profit percent",(postr/len(sl))*100)
    st.write("Max Profit",maxprofit(sl,bl,q))
    st.write("Max Loss",maxloss(sl,bl,q))
    st.write("Total Loss",totalloss(sl,bl,q))
    st.write("Total Profit",totalprofit(sl,bl,q))
    st.write("Average Loss",avgloss(sl,bl,q))
    st.write("Average Loss Percent",avglossper(sl,bl,q))
    st.write("Max Consecutive Loss",maxcontloss(sl,bl))
    st.write("Max Consecutive Profit",maxcontprofit(sl,bl))
#     st.write("Max Consecutive Loss",maxcontloss(sl,bl))
    st.write("Largest Profit",largestprofit(sl,bl,q))
    st.write("Largest Loss",largestloss(sl,bl,q))
    st.write("Max Trade Drawdown",maxdrawdown(data))
    st.write("Max System Drawdown",max_drawdown_system(data['Close']))
    if totalloss(sl,bl,q)>0:
        st.write("Profit Factor",abs(totalprofit(sl,bl,q)/totalloss(sl,bl,q)))
    if avgloss(sl,bl,q)>0:
        st.write("Payoff Ratio",abs(avp/avgloss(sl,bl,q)))
#     st.write("Here are the trades which happened on the historical data with your strategy and parametrs",trades(sl,bl,data))
    df = pd.DataFrame()
    df['Bought at']=bl
    df=df[:len(sl)]
    df['Sold at']=sl
    st.write("Here are the trades which happened on the historical data with your strategy and parametrs")
    df

def emacrossover3(data,fast1,fast2,fast3,slow,init_cash):
    data['fast1'] = ema(data,fast1)
    data['fast2'] = ema(data, fast2)
    data['fast3'] = ema(data,fast3)
    data['slow'] = ema(data, slow)
    flag=0
    buy=0
    pp=0
    bl=[]
    sl=[]
    q=[]
    trades=0
#     init_cash=100000
    # shares=portfolio_size/data['Close'].mean()
    df=data
    df['Signal'] = 0.0
    for i in range(len(data)):
        if data['fast1'][i]>data['slow'][i] and flag==0 and data['fast2'][i]>data['slow'][i] and data['fast3'][i]>data['slow'][i]:
#             print("Buy at ")
#             print(data['Close'][i])
            buy=data['Close'][i]
            q.append(init_cash/buy)
            bl.append(data['Close'][i])
            trades=trades+1
            flag=1
            df['Signal'] =1
        if data['fast1'][i]<data['slow'][i] and flag==1 and data['fast2'][i]<data['slow'][i] and data['fast3'][i]<data['slow'][i] or (data['Close'][i]<0.85*buy and flag==1):
#             print("sell at ")
#             print(data['Close'][i])
            sl.append(data['Close'][i])
            trades=trades+1
            flag=0
    fp=0
    avp=0
    postr=1
    for i in range(len(bl)-1):
        fp=fp+(sl[i]-bl[i])*q[i]
        if sl[i]-bl[i]>0:
            avp=avp+(sl[i]-bl[i])*q[i]
            postr=postr+1
    avp=avp/postr
    fast1 = ema(data,fast1)
    fast2 = ema(data, fast2)
    fast3 = ema(data,fast3)
    slow = ema(data, slow)
    plt.figure(figsize=(20,12))
    plt.plot(data.Close)
    plt.plot(fast1,'r')
    plt.plot(fast2,'g')
    plt.plot(fast3,'y')
    plt.plot(slow,'b')
    st.pyplot(fig=None, clear_figure=None)
    df['Position'] = df['Signal'].diff()
    plt.figure(figsize = (20,10))
    df['Close'].plot(color = 'k', label= 'Close Price') 
    df['fast1'].plot(color = 'b',label = 'fast1 SMA') 
    df['fast2'].plot(color = 'y',label = 'fast2 SMA')
    df['fast3'].plot(color = 'r',label = 'fast3 SMA')
    df['slow'].plot(color = 'g', label = 'slow SMA')

    plt.plot(df[df['Position'] == 1].index, 
             df['fast3'][df['Position'] == 1], 
             '^', markersize = 15, color = 'g', label = 'buy')

    plt.plot(df[df['Position'] == -1].index, 
             df['fast3'][df['Position'] == -1], 
             'v', markersize = 15, color = 'r', label = 'sell')
    plt.ylabel('Price in Rupees', fontsize = 15 )
    plt.xlabel('Date', fontsize = 15 )
    plt.title('MAC', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()
    st.pyplot(fig=None, clear_figure=None)
    st.write("Profit ",fp)
    st.write("Profit Percent",(fp/init_cash)*100)
    sharpe=sharpe_ratio(data['Close'])
    st.write("Sharpe Ratio",sharpe)
    st.write("Average Profit",avp)
    if len(sl)>0:
        st.write("Average profit percent",(postr/len(sl))*100)
    st.write("Max Profit",maxprofit(sl,bl,q))
    st.write("Max Loss",maxloss(sl,bl,q))
    st.write("Total Loss",totalloss(sl,bl,q))
    st.write("Total Profit",totalprofit(sl,bl,q))
    st.write("Average Loss",avgloss(sl,bl,q))
    st.write("Average Loss Percent",avglossper(sl,bl,q))
    st.write("Max Consecutive Loss",maxcontloss(sl,bl))
    st.write("Max Consecutive Profit",maxcontprofit(sl,bl))
#     st.write("Max Consecutive Loss",maxcontloss(sl,bl))
    st.write("Largest Profit",largestprofit(sl,bl,q))
    st.write("Largest Loss",largestloss(sl,bl,q))
    st.write("Max Trade Drawdown",maxdrawdown(data))
    st.write("Max System Drawdown",max_drawdown_system(data['Close']))
    if totalloss(sl,bl,q)>0:
        st.write("Profit Factor",abs(totalprofit(sl,bl,q)/totalloss(sl,bl,q)))
    if avgloss(sl,bl,q)>0:
        st.write("Payoff Ratio",abs(avp/avgloss(sl,bl,q)))
#     st.write("Here are the trades which happened on the historical data with your strategy and parametrs",trades(sl,bl,data))
    df = pd.DataFrame()
    df['Bought at']=bl
    df=df[:len(sl)]
    df['Sold at']=sl
    st.write("Here are the trades which happened on the historical data with your strategy and parametrs")
    df
    
def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df
    
def plot_macd(prices, macd, signal, hist):
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(prices)
    ax2.plot(macd, color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(signal, color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax2.bar(prices.index[i], hist[i], color = '#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], color = '#26a69a')

    plt.legend(loc = 'lower right')

    plot_macd(googl['close'], googl_macd['macd'], googl_macd['signal'], googl_macd['hist'])
    st.pyplot(fig=None, clear_figure=None)

     
    
# def bollinger_bands_fibonacci_strategy(df: pd.DataFrame, length: int, stdDev: float) -> pd.DataFrame:
#     # Calculate Bollinger Bands
#     length = 20
#     stdDev = 2

#     # Calculate Bollinger Bands
#     data['sma'] = data['Close'].rolling(length).mean()
#     data['std_dev'] = data['Close'].rolling(length).std()
#     data['bb_lower'] = data['sma'] - stdDev * data['std_dev']
#     data['bb_middle'] = data['sma']
#     data['bb_upper'] = data['sma'] + stdDev * data['std_dev']

#     # Calculate Fibonacci retracement levels
#     high = data['High'].max()
#     low = data['Low'].min()
#     data['fib_23.6'] = low + (high - low) * 23.6 / 100
#     data['fib_38.2'] = low + (high - low) * 38.2 / 100
#     data['fib_50.0'] = low + (high - low) * 50.0 / 100
#     data['fib_61.8'] = low + (high - low) * 61.8 / 100
#     data['fib_100.0'] = low + (high - low) * 100.0 / 100

#     # Initialize variables
#     profit = 0
#     position = "none"

#     # Iterate through data
#     for index, row in data.iterrows():
#         # Check for long entry
#         if position == "none" and row['Close'] < row['bb_lower'] and row['Close'] < row['fib_23.6']:
#             position = "long"
#             entry_price = row['Close']
#         # Check for long exit
#         elif position == "long" and (row['Close'] > row['bb_upper'] or row['Close'] > row['fib_61.8']):
#             position = "none"
#             profit += row['Close'] - entry_price
#         # Check for short entry
#         elif position == "none" and row['Close'] > row['bb_upper'] and row['Close'] > row['fib_61.8']:
#             position = "short"
#             entry_price = row['Close']
#         # Check for short exit
#         elif position == "short" and (row['Close'] < row['bb_lower'] or row['Close'] < row['fib_23.6']):
#             position = "none"
#             profit += entry_price - row['Close']

#     print("Profit: ",profit)


# In[116]:


def macd(price,slow,fast,smooth):
    def get_macd(price, slow, fast, smooth):
        exp1 = price.ewm(span = fast, adjust = False).mean()
        exp2 = price.ewm(span = slow, adjust = False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
        signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
        hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
        frames =  [macd, signal, hist]
        df = pd.concat(frames, join = 'inner', axis = 1)
        return df
    rmacd = get_macd(data['Close'], slow, fast, smooth)
    df1=rmacd
    data['macd']=df1['macd']
    data['signal']=df1['signal']
    data['hist']=df1['hist']
    
    def plot_macd(prices, macd, signal, hist):
        ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

        ax1.plot(prices)
        ax2.plot(macd, color = 'grey', linewidth = 1.5, label = 'MACD')
        ax2.plot(signal, color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

        for i in range(len(prices)):
            if str(hist[i])[0] == '-':
                ax2.bar(prices.index[i], hist[i], color = '#ffffff')
            else:
                ax2.bar(prices.index[i], hist[i], color = '#26a69a')

        plt.legend(loc = 'lower right')
    plot_macd(data['Close'], data['macd'], data['signal'], data['hist'])
    st.pyplot(fig=None, clear_figure=None)
    def implement_macd_strategy(prices, data):    
        buy_price = []
        sell_price = []
        macd_signal = []
        signal = 0

        for i in range(len(data)):
            if data['macd'][i] > data['signal'][i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)
            elif data['macd'][i] < data['signal'][i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        return buy_price, sell_price, macd_signal
            
    buy_price, sell_price, macd_signal = implement_macd_strategy(data['Close'], data)
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(data['Close'], color = 'skyblue', linewidth = 2, label = 'Ticker')
    ax1.plot(data.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax1.plot(data.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax1.legend()
    ax1.set_title('MACD SIGNALS')
    ax2.plot(data['macd'], color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(data['signal'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(data)):
        if str(data['hist'][i])[0] == '-':
            ax2.bar(data.index[i], data['hist'][i], color = '#ef5350')
        else:
            ax2.bar(data.index[i], data['hist'][i], color = '#26a69a')

    plt.legend(loc = 'lower right')
    plt.show()
    st.pyplot(fig=None, clear_figure=None)
    position = []
    for i in range(len(macd_signal)):
        if macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)

    for i in range(len(data['Close'])):
        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]

    macd = data['macd']
    signal = data['signal']
    close_price = data['Close']
    macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(data.index)
    position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(data.index)

    frames = [close_price, macd, signal, macd_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis = 1)

    strategy
    from math import floor
    import pyopencl as cl
    ret = pd.DataFrame(np.diff(data['Close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(ret)):
        try:
            returns = ret['returns'][i]*strategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass

    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    investment_value = 100000
    number_of_stocks = floor(investment_value/data['Close'][0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks*macd_strategy_ret_df['macd_returns'][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    st.write("Total Investment Return",total_investment_ret)
    st.write('Profit percentage of the MACD strategy :',profit_percentage)
    sharpe=sharpe_ratio(price['Close'])
    st.write("Sharpe Ratio",sharpe)


# In[133]:


import numpy as np
import pandas as pd

def bollinger_bands_fibonacci_strategy(data,length,stdDev,init_cash):
    # Bollinger Bands parameters
    length = length
    stdDev = stdDev
    q=[]
    # Calculate Bollinger Bands
    data['sma'] = data['Close'].rolling(length).mean()
    data['std_dev'] = data['Close'].rolling(length).std()
    data['bb_lower'] = data['sma'] - stdDev * data['std_dev']
    data['bb_middle'] = data['sma']
    data['bb_upper'] = data['sma'] + stdDev * data['std_dev']

    # Calculate Fibonacci retracement levels
    high = data['High'].max()
    low = data['Low'].min()
    data['fib_23.6'] = low + (high - low) * 23.6 / 100
    data['fib_38.2'] = low + (high - low) * 38.2 / 100
    data['fib_50.0'] = low + (high - low) * 50.0 / 100
    data['fib_61.8'] = low + (high - low) * 61.8 / 100
    data['fib_100.0'] = low + (high - low) * 100.0 / 100

    # Initialize variables
    profit = 0
    q1=0
    position = "none"
    long_entry_signals = []
    long_exit_signals = []
    short_entry_signals = []
    short_exit_signals = []
    # Iterate through data
    for index, row in data.iterrows():
        # Check for long entry
        if position == "none" and row['Close'] < row['bb_lower'] and row['Close'] < row['fib_23.6']:
            position = "long"
            entry_price = row['Close']
            q.append(init_cash/row['Close'])
            q1=init_cash/row['Close']
            long_entry_signals.append(row['Close'])
        # Check for long exit
        elif position == "long" and (row['Close'] > row['bb_upper'] or row['Close'] > row['fib_61.8']):
            position = "none"
            profit += (row['Close'] - entry_price)*q1
            long_exit_signals.append(row['Close'])
        # Check for short entry
        elif position == "none" and row['Close'] > row['bb_upper'] and row['Close'] > row['fib_61.8']:
            position = "short"
            entry_price = row['Close']
            q1=init_cash/row['Close']
            short_entry_signals.append(row['Close'])
        # Check for short exit
        elif position == "short" and (row['Close'] < row['bb_lower'] or row['Close'] < row['fib_23.6']):
            position = "none"
            profit += (entry_price - row['Close'])*q1
            short_exit_signals.append(row['Close'])

    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['bb_lower'], label='Lower Bollinger Band', linestyle='dashed')
    plt.plot(data['bb_upper'], label='Upper Bollinger Band', linestyle='dashed')
    plt.plot(data['fib_23.6'], label='23.6% Fibonacci Retracement', linestyle='dashed')
    plt.plot(data['fib_61.8'], label='61.8% Fibonacci Retracement', linestyle='dashed')
    
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig=None, clear_figure=None)
    st.write("Profit ",profit)
    st.write("Profit Percent",(profit/init_cash)*100)
    sharpe=sharpe_ratio(data['Close'])
    st.write("Sharpe Ratio",sharpe)
#     st.write("Profit: ",profit)
    st.write("Max Profit",maxprofit(long_exit_signals,long_entry_signals,q))
    st.write("Max Loss",maxloss(long_exit_signals,long_entry_signals,q))
    st.write("Total Loss",totalloss(long_exit_signals,long_entry_signals,q))
    st.write("Total Profit",totalprofit(long_exit_signals,long_entry_signals,q))
    st.write("Average Loss",avgloss(long_exit_signals,long_entry_signals,q))
    st.write("Average Loss Percent",avglossper(long_exit_signals,long_entry_signals,q))
    st.write("Max Consecutive Loss",maxcontloss(long_exit_signals,long_entry_signals))
    st.write("Max Consecutive Profit",maxcontprofit(long_exit_signals,long_entry_signals))
    st.write("Max Consecutive Loss",maxcontloss(long_exit_signals,long_entry_signals))
    st.write("Largest Profit",largestprofit(long_exit_signals,long_entry_signals,q))
    st.write("Largest Loss",largestloss(long_exit_signals,long_entry_signals,q))
    st.write("Max Trade Drawdown",maxdrawdown(data))
    st.write("Max System Drawdown",max_drawdown_system(data['Close']))
    if totalloss(long_exit_signals,long_entry_signals,q)>0:
        st.write("Profit Factor",abs(totalprofit(long_exit_signals,long_entry_signals,q)/totalloss(long_exit_signals,long_entry_signals,q)))
#     st.write("Payoff Ratio",abs(avp/avgloss(long_exit_signals,long_entry_signals,q)))
    df = pd.DataFrame()
    df['Long Trade Bought at']=long_entry_signals
    df=df[:len(long_exit_signals)]
    df['Long Trade Sold at']=long_exit_signals
    df1 = pd.DataFrame()
    df1['Short Trade Bought at']=short_entry_signals
    df1=df1[:len(short_exit_signals)]
    df1['Short Trade Sold at']=short_exit_signals
    st.write("Long Trades Happened")
    df
    st.write("Short Trades Happened")
    df1
    


# In[ ]:





# In[118]:


# bb(data,20,14)


# In[119]:


# emacrossover3(data,20,50,100,200)


# In[120]:


#Plot gaphs too now


# In[121]:


def main(): 
    activities=["Moving Average Crossover","Bollinger Bands","Bollinge Bands with Fibonacci Retracement","4 Ema Crossover","MACD"]
    choice=st.sidebar.selectbox("Select Strategy",activities)
    if choice=="Moving Average Crossover":
        fast=st.sidebar.slider("fast_period", min_value=0, max_value=200, value=50)
        slow=st.sidebar.slider("slow_period", min_value=fast+1, max_value=500, value=200)
        init_cash=st.sidebar.slider("Initial_Cash", min_value=1000, max_value=10000000, value=100000)
        st.write(macrossover(data,fast,slow,init_cash,0,0))
#         st.write("YO")
    if choice=="Bollinger Bands":
        period=st.sidebar.slider("Period", min_value=0, max_value=500, value=20)
        rsi=st.sidebar.slider("RSI_period", min_value=0, max_value=500, value=14)
        rsis=st.sidebar.slider("RSI_Oversold_threshold", min_value=0, max_value=100, value=40)
        rsib=st.sidebar.slider("RSI_Overbought_threshold", min_value=rsis, max_value=100, value=60)
        init_cash=st.sidebar.slider("Initial_Cash", min_value=1000, max_value=10000000, value=100000)
        st.write(bb(data,period,rsi,rsis,rsib,init_cash))
    if choice=="Bollinge Bands with Fibonacci Retracement":
        period=st.sidebar.slider("BB Length", min_value=0, max_value=500, value=20)
        std=st.sidebar.slider("Standard Deviation", min_value=0, max_value=500, value=2)
        init_cash=st.sidebar.slider("Initial_Cash", min_value=1000, max_value=10000000, value=100000)
        st.write(bollinger_bands_fibonacci_strategy(data,period,std,init_cash))
    if choice=="4 Ema Crossover":
        fast1=st.sidebar.slider("fast_period 1", min_value=0, max_value=50, value=8)
        fast2=st.sidebar.slider("fast_period 2", min_value=fast1+1, max_value=100, value=13)
        fast3=st.sidebar.slider("fast_period 3", min_value=fast2+1, max_value=200, value=21)
        slow=st.sidebar.slider("slow_period", min_value=fast3+1, max_value=500, value=55)
        init_cash=st.sidebar.slider("Initial_Cash", min_value=1000, max_value=10000000, value=100000)
        st.write(emacrossover3(data,fast1,fast2,fast3,slow,init_cash))
    if choice=="MACD":
        fast=st.sidebar.slider("fast_period", min_value=1, max_value=200, value=12)
        slow=st.sidebar.slider("slow_period", min_value=fast+1, max_value=300, value=26)
        smooth=st.sidebar.slider("smooth", min_value=1, max_value=200, value=9)
        st.write(macd(data,slow,fast,smooth))
    
if __name__=='__main__':
    main()


# In[81]:


# data


# In[128]:


# bollinger_bands_fibonacci_strategy(data,20,3,100000)


# In[29]:


# import numpy as np
# import pandas as pd

# # Load historical data


# # Bollinger Bands parameters
# length = 20
# stdDev = 2

# # Calculate Bollinger Bands
# data['sma'] = data['Close'].rolling(length).mean()
# data['std_dev'] = data['Close'].rolling(length).std()
# data['bb_lower'] = data['sma'] - stdDev * data['std_dev']
# data['bb_middle'] = data['sma']
# data['bb_upper'] = data['sma'] + stdDev * data['std_dev']

# # Initialize variables
# profit = 0
# position = "none"

# # Iterate through data
# for index, row in data.iterrows():
#     # Check for long entry
#     if position == "none" and row['Close'] < row['bb_lower']:
#         position = "long"
#         entry_price = row['Close']
#     # Check for long exit
#     elif position == "long" and row['Close'] > row['bb_upper']:
#         position = "none"
#         profit += row['Close'] - entry_price
#     # Check for short entry
#     elif position == "none" and row['Close'] > row['bb_upper']:
#         position = "short"
#         entry_price = row['Close']
#     # Check for short exit
#     elif position == "short" and row['Close'] < row['bb_lower']:
#         position = "none"
#         profit += entry_price - row['Close']

# print("Profit: ",profit)


# In[75]:


# import numpy as np
# import pandas as pd

# def bollinger_bands_fibonacci_strategy(data,length,stdDev):
#     # Bollinger Bands parameters
#     length = length
#     stdDev = stdDev

#     # Calculate Bollinger Bands
#     data['sma'] = data['Close'].rolling(length).mean()
#     data['std_dev'] = data['Close'].rolling(length).std()
#     data['bb_lower'] = data['sma'] - stdDev * data['std_dev']
#     data['bb_middle'] = data['sma']
#     data['bb_upper'] = data['sma'] + stdDev * data['std_dev']

#     # Calculate Fibonacci retracement levels
#     high = data['High'].max()
#     low = data['Low'].min()
#     data['fib_23.6'] = low + (high - low) * 23.6 / 100
#     data['fib_38.2'] = low + (high - low) * 38.2 / 100
#     data['fib_50.0'] = low + (high - low) * 50.0 / 100
#     data['fib_61.8'] = low + (high - low) * 61.8 / 100
#     data['fib_100.0'] = low + (high - low) * 100.0 / 100

#     # Initialize variables
#     profit = 0
#     position = "none"
#     long_entry_signals = []
#     long_exit_signals = []
#     short_entry_signals = []
#     short_exit_signals = []
#     # Iterate through data
#     for index, row in data.iterrows():
#         # Check for long entry
#         if position == "none" and row['Close'] < row['bb_lower'] and row['Close'] < row['fib_23.6']:
#             position = "long"
#             entry_price = row['Close']
#             long_entry_signals.append(row['Close'])
#         # Check for long exit
#         elif position == "long" and (row['Close'] > row['bb_upper'] or row['Close'] > row['fib_61.8']):
#             position = "none"
#             profit += row['Close'] - entry_price
#             long_exit_signals.append(row['Close'])
#         # Check for short entry
#         elif position == "none" and row['Close'] > row['bb_upper'] and row['Close'] > row['fib_61.8']:
#             position = "short"
#             entry_price = row['Close']
#             short_entry_signals.append(row['Close'])
#         # Check for short exit
#         elif position == "short" and (row['Close'] < row['bb_lower'] or row['Close'] < row['fib_23.6']):
#             position = "none"
#             profit += entry_price - row['Close']
#             short_exit_signals.append(row['Close'])

#     plt.plot(data['Close'], label='Close Price')
#     plt.plot(data['bb_lower'], label='Lower Bollinger Band', linestyle='dashed')
#     plt.plot(data['bb_upper'], label='Upper Bollinger Band', linestyle='dashed')
#     plt.plot(data['fib_23.6'], label='23.6% Fibonacci Retracement', linestyle='dashed')
#     plt.plot(data['fib_61.8'], label='61.8% Fibonacci Retracement', linestyle='dashed')

#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()

#     print("Profit: ",profit)


# In[76]:


# plt.scatter(long_entry_signals, np.zeros_like(long_entry_signals) + data['Close'].min(), color='g', label='Buy Signals')
# plt.scatter(short_entry_signals, np.zeros_like(short_entry_signals) + data['Close'].max(), color='r', label='Sell Signals')


# In[77]:


# bollinger_bands_fibonacci_strategy(data,20,2)


# In[78]:


# plt.scatter(long_entry_signals, np.zeros_like(long_entry_signals) + 23.6, color='g', label='Long Signals')
# plt.scatter(short_entry_signals, np.zeros_like(short_entry_signals) + 61.8, color='r', label='Short Signals')


# In[79]:


#     df=data
#     df['Signal'] = np.where(row['Close'] < row['bb_lower'] and row['Close'] < row['fib_23.6'], 1.0)
#     df['Signal'] = np.where(row['Close'] > row['bb_upper'] or row['Close'] > row['fib_61.8'], 0)
#     df['Position'] = df['Signal'].diff()
#     plt.figure(figsize = (20,10))
#     df['Close'].plot(color = 'k', label= 'Close Price') 
#     df['fast'].plot(color = 'b',label = 'fast SMA') 
#     df['slow'].plot(color = 'g', label = 'slow SMA')

#     plt.plot(df[df['Position'] == 1].index, 
#              df['fast'][df['Position'] == 1], 
#              '^', markersize = 15, color = 'g', label = 'buy')

#     plt.plot(df[df['Position'] == -1].index, 
#              df['fast'][df['Position'] == -1], 
#              'v', markersize = 15, color = 'r', label = 'sell')


# In[ ]:




