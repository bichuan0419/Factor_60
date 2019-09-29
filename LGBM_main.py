import pandas as pd
import os, math
import numpy as np
from datetime import timedelta
import seaborn as sns
#Set color palette for graphs
sns.set_palette(sns.color_palette('hls', 7))
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm, skew #for some statistics
import time
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

from prepare_data import prepare_data
from utilities import filter_df_by_date, pick_factor_within_period, Stock_to_buy, \
    CalcReturns,max_drawdown, Stock_to_buy_LGBM


New_Factor_df_date = pd.read_csv('New_Factor_df_date.csv',index_col=0)
Idxrtn_series= pd.read_csv('Idxrtn_series.csv',index_col=0)
# All stocks as all_stkid
all_stkid = New_Factor_df_date['stkcd'].unique()
New_Factor_df_date['stkcd'] = New_Factor_df_date['stkcd'].astype(str)
all_stkid = [str(x) for x in all_stkid]
New_Factor_df_date.index = pd.to_datetime(New_Factor_df_date.index)
New_Factor_df_date.loc[New_Factor_df_date['fret']> 0.1,'fret'] = 0.1
New_Factor_df_date.loc[New_Factor_df_date['fret']< -0.1,'fret'] = -0.1

# select dataframe from original data
df = filter_df_by_date(startdate='2016-03-01', enddate='2018-04-30', df=New_Factor_df_date)
# df=New_Factor_df_date

# try a smaller sample to test
# df = df.iloc[0:,0:30]
prev_stock = None
curr_date = None
Total_Return = []
Timeline = []
temp_total = 1
train_days = 20

for i in range(len(df.index.unique()) - train_days):
    # setup starting and ending date for training purposes
    startdate = df.index.unique()[i]

    testdate = df.index.unique()[i + train_days]

    # get features from the past training period

    curr_df = filter_df_by_date(startdate=startdate, enddate=testdate, df=df)
    stock_to_buy = Stock_to_buy_LGBM(curr_df, all_stkid=all_stkid, percentile=10)

    temp_total *= (1 + CalcReturns(stock_to_buy, currdate=testdate, New_Factor_df_date=New_Factor_df_date,
                                   Idxrtn_series=Idxrtn_series,
                                   all_stkid=all_stkid, prev_stock=prev_stock))
    print('test date: ', testdate, ' with return ', temp_total)
    Total_Return.append(temp_total)
    Timeline.append(testdate)
    prev_stock = stock_to_buy

# Timeline = New_Factor_df_date.index.unique()[New_Factor_df_date.index.unique() >= ]
print('Maxdrawdown: ', max_drawdown(Total_Return))
print('Market Cap: ', Total_Return[-1])
plt.plot(Timeline, Total_Return)
fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.show()