# %%
import requests 
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# %%
from datetime import datetime
"""
var StaticTreshholdData=[[]];
var ClosingPriceData=[[]];
var InstrumentStateData=[[20210330,1,'A ']];
var IntraTradeData=[[ ]];
var ShareHolderData=[[ ]];
var ClientTypeData= [];
var BestLimitData=[[ ]];

"""
url = 'http://cdn.tsetmc.com/Loader.aspx?ParTree=15131P&i=778253364357513&d=20210502#url'
data_date = re.search(r'&d=(\d+)#url', url).group(1)
data_date

# %%
r = requests.get(url)
#%%
r.encoding
# %%
IntraTradeData = re.findall(r'var IntraTradeData=\[\[(.+)\]\];', r.text)[0].split('],[')

# %%
tmp = []
for element in IntraTradeData:
    tmp.append([items.replace("'", "") for items in element.split(',')])

intra_trade_df = pd.DataFrame(tmp, columns=['id','time','volum','price','deleted'])
intra_trade_df['time'] = intra_trade_df['time'] + '-' + data_date
intra_trade_df['time'] = pd.to_datetime(intra_trade_df['time'],format='%H:%M:%S-%Y%m%d')
intra_trade_df
# %%

fig, ax = plt.subplots(1,1)
ax.set(title='volum')
ax.grid()
ax.hist(intra_trade_df['volum'].astype(np.float).values, bins=200)

plt.show()
# %%
fig, ax = plt.subplots(1,1)
ax.set(title='price')
ax.grid()
ax.hist(intra_trade_df['price'].astype(np.float).values, bins=500)

plt.show()
# %%
intra_trade_df.info()
# %%
intra_trade_df.describe()
# %%
intra_trade_df.dtypes
# %%
StaticTreshholdData = re.findall(r'var StaticTreshholdData=\[\[(.+)\]\];', r.text)[0].split('],[')
StaticTreshholdData
# %%
ClosingPriceData = re.findall(r'var ClosingPriceData=\[\[(.+)\]\];', r.text)[0].split('],[')
CPD_tmp =[]
for item in ClosingPriceData:
    CPD_tmp.append([i.replace("'", "") for i in item.split(',')])

ClosingPriceData_df = pd.DataFrame(CPD_tmp)
ClosingPriceData_df
# %%
InstrumentStateData = re.findall(r'var InstrumentStateData=\[\[(.+)\]\];', r.text)[0].split('],[')
InstrumentStateData
# %%
ShareHolderData = re.findall(r'var ShareHolderData=\[\[(.+)\]\];', r.text)[0].split('],[')
ShareHolderData
# %%
ClientTypeData = re.findall(r'var ClientTypeData=\[(.+)\];', r.text)[0].split('],[')
ClientTypeData
# %%
BestLimitData = re.findall(r'var BestLimitData=\[\[(.+)\]\];', r.text)[0].split('],[')
BLD_tmp =[]
for item in BestLimitData:
    BLD_tmp.append([i.replace("'", "") for i in item.split(',')])

BestLimitData_df = pd.DataFrame(BLD_tmp, columns=['time', 'position', 'buyers_num', 'buy_volume','buy_price','sell_price','sell_volume','sellers_num'])
BestLimitData_df['time'] = BestLimitData_df['time'] + '-' + data_date
BestLimitData_df['time'] = pd.to_datetime(BestLimitData_df['time'],format='%H%M%S-%Y%m%d')

BestLimitData_df.iloc[-4500:-4450,:]
# %%
