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
from utilities import filter_df_by_date, pick_factor_within_period, Stock_to_buy_DOW, CalcReturns,max_drawdown

# New_Factor_df_date, Idxrtn_series= prepare_data()
# New_Factor_df_date.to_csv('New_Factor_df_date.csv')
# Idxrtn_series.to_csv('Idxrtn_series.csv')


New_Factor_df_date = pd.read_csv('New_Factor_df_date.csv',index_col=0)
Idxrtn_series= pd.read_csv('Idxrtn_series.csv',index_col=0)


# All stocks as all_stkid
all_stkid = New_Factor_df_date['stkcd'].unique()
New_Factor_df_date['stkcd'] = New_Factor_df_date['stkcd'].astype(str)
all_stkid = [str(x) for x in all_stkid]
New_Factor_df_date.index = pd.to_datetime(New_Factor_df_date.index)

# select dataframe from original data
df = filter_df_by_date(startdate='2016-03-01',enddate='2019-04-30',df=New_Factor_df_date)
# df=New_Factor_df_date

# try a smaller sample to test
# df = df.iloc[0:,0:30]
prev_stock = None
curr_date = None
Total_Return = [1]
Timeline = []
temp_total = 1
train_days = 30
selected_features = ['tot_rank29']

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
    # print(selected_features)

    curr_df = df[df.index == testdate]

    # Stock_to_buy(factors=selected_features, curr_df=curr_df, all_stkid=all_stkid, percentile=10)
    stock_to_buy = Stock_to_buy_DOW(factors=selected_features, curr_df=curr_df, all_stkid=all_stkid, prev_stock=prev_stock, dow=4, percentile=10)

    # num_stock_to_buy = 0
    # for k,v in stock_to_buy.items():
    #     if v != 0:
    #         num_stock_to_buy += 1
    # print(num_stock_to_buy)

    #     print(temp_total)
    temp_total *= (1 +CalcReturns(stock_to_buy, currdate=testdate, New_Factor_df_date=New_Factor_df_date, Idxrtn_series=Idxrtn_series,
                all_stkid=all_stkid, prev_stock=prev_stock))

    log_file.update({testdate: temp_total})

    Total_Return.append(temp_total)
    print('test date: ', testdate, ' with return: ', temp_total, ' current max_drawdown: ', max_drawdown(Total_Return))

    Timeline.append(testdate)
    prev_stock = stock_to_buy

# Print something
print('Maxdrawdown: ', max_drawdown(Total_Return))
print('Market Cap: ', Total_Return[-1])

# save results
currFolder = os.getcwd()
log_file_name = selected_features[0] + '_log_file_DOW.csv'
path_to_add = os.path.join(os.path.join(currFolder, 'log_data/',),log_file_name)
log_file = pd.Series(log_file)
log_file.to_csv(path_to_add)

# Plot something
plt.plot(Timeline, Total_Return[1:])
fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.show()
