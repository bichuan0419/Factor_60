import pandas as pd
import os, math
import numpy as np
from datetime import timedelta
import seaborn as sns
#Set color palette for graphs
sns.set_palette(sns.color_palette('hls', 7))
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from prepare_data import prepare_data
from utilities import filter_df_by_date, pick_factor_within_period, Stock_to_buy,\
    CalcReturns,max_drawdown,vote_for_stocks_DOW,vote_for_stocks

# New_Factor_df_date, Idxrtn_series= prepare_data()
# New_Factor_df_date.to_csv('New_Factor_df_date.csv')
# Idxrtn_series.to_csv('Idxrtn_series.csv')

# Always needed!
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
df = filter_df_by_date(startdate='2016-03-01',enddate='2019-04-30',df=New_Factor_df_date)
# df=New_Factor_df_date

# try a smaller sample to test
prev_stock = {all_stkid[i]: 0 for i in range(len(all_stkid))}
curr_date = None
Total_Return = [1]
Timeline = []
temp_total = 1
train_days = 30

# Create log file
log_file = {}

for i in range(len(df.index.unique()) - train_days):
    # setup starting and ending date for training purposes
    startdate = df.index.unique()[i]
    enddate = df.index.unique()[i + train_days - 1]
    # get test date
    testdate = df.index.unique()[i + train_days]

    # get features from the past training period
    # selected_features = pick_factor_within_period(startdate, enddate, df, New_Factor_df_date, Idxrtn_series, all_stkid, number_factors=3)

    curr_df = df[df.index == testdate]

    stock_to_buy = vote_for_stocks(df=curr_df, testdate=testdate, all_stkid=all_stkid,
                                       percentile=10)
    # stock_to_buy = vote_for_stocks_DOW(df=curr_df,testdate=testdate,all_stkid=all_stkid,prev_stock = prev_stock, dow=3,percentile=10)

    # temp_ret ~= drawdown if it is negative
    temp_ret = CalcReturns(stock_to_buy, currdate=testdate, New_Factor_df_date=New_Factor_df_date, Idxrtn_series=Idxrtn_series,
                all_stkid=all_stkid, prev_stock=prev_stock)
    temp_total *= (1 + temp_ret)

    log_file.update({testdate:temp_total})
    Total_Return.append(temp_total)
    print('test date: ', testdate, ' with return: ', temp_total, ' current max_drawdown: ', max_drawdown(Total_Return))
    Timeline.append(testdate)
    prev_stock = stock_to_buy

# Timeline = New_Factor_df_date.index.unique()[New_Factor_df_date.index.unique() >= ]
print('Maxdrawdown: ', max_drawdown(Total_Return))
print('Market Cap: ', Total_Return[-1])
log_file = pd.Series(log_file)
log_file.to_csv('log_file.csv')
plt.plot(Timeline, Total_Return[1:])
fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.show()


