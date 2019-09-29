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
from utilities import filter_df_by_date, pick_factor_within_period, Stock_to_buy, CalcReturns, max_drawdown

# New_Factor_df_date, Idxrtn_series= prepare_data()
# New_Factor_df_date.to_csv('New_Factor_df_date.csv')
# Idxrtn_series.to_csv('Idxrtn_series.csv')

# Load data
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


# initialization
train_days = 0
feature_columns = df.columns[2:]
summary_df = pd.DataFrame({'Factors': feature_columns,
                          'Max_drawdown':0,
                          'Market Cap':0})
summary_df.set_index('Factors',inplace =True)


for selected_features in feature_columns:
    # initialize for each feature
    selected_features = [selected_features]
    Total_Return = []
    Timeline = []
    prev_stock = {all_stkid[i]: 0 for i in range(len(all_stkid))}
    temp_total = 1
    # Create log file
    log_file = {}
    for i in range(len(df.index.unique()) - train_days):

        # get test date
        testdate = df.index.unique()[i + train_days]

        # # get features
        # selected_features = ['tot_rank1']

        curr_df = df[df.index == testdate]

        stock_to_buy = Stock_to_buy(factors=selected_features, curr_df=curr_df, all_stkid=all_stkid, percentile=10)
        #     print(temp_total)
        ret_rate = CalcReturns(stock_to_buy, currdate=testdate, New_Factor_df_date=New_Factor_df_date, Idxrtn_series=Idxrtn_series,
                    all_stkid=all_stkid, prev_stock=prev_stock)
        temp_total *= (1 + ret_rate)
        log_file.update({testdate: ret_rate})
        # print('test date: ', testdate, ' with return ', temp_total)
        Total_Return.append(temp_total)
        Timeline.append(testdate)
        prev_stock = stock_to_buy
    # print('Factor: ', selected_features, ' Maxdrawdown: ', max_drawdown(Total_Return), ' Profit: ', Total_Return[-1])
    summary_df.loc[selected_features, ['Max_drawdown', 'Market Cap']] = [max_drawdown(Total_Return), Total_Return[-1]]
    # save results
    currFolder = os.getcwd()
    log_file_name = selected_features[0] + '_log_file.csv'
    path_to_add = os.path.join(os.path.join(currFolder, 'log_data/', ), log_file_name)
    log_file_save = pd.Series(log_file)
    log_file_save.to_csv(path_to_add)

print(summary_df)
summary_df.to_csv('Three_year_summary_df.csv')

input("Press Enter to continue...")
