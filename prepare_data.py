import pandas as pd
import os, math
import numpy as np
from datetime import timedelta
import seaborn as sns
#Set color palette for graphs
sns.set_palette(sns.color_palette('hls', 7))
import warnings
warnings.filterwarnings("ignore")


def prepare_data():
    currFolder = os.getcwd()
    Factors60_path = os.path.join(currFolder, 'data/Factors_60.csv')
    Index0621_path = os.path.join(currFolder, 'data/index_0619.csv')
    # Factors_df: Dataframe of factors and stock returns
    Factors_df = pd.read_csv(Factors60_path)
    # Index_df = Index return (not sure what that means, NEED TO LOOK UP!)
    Index_df = pd.read_csv(Index0621_path)

    # Aligning formats: date or numerical values
    Factors_df['trddt'] = pd.to_datetime(Factors_df['trddt'])
    Factors_df['stkcd'] = Factors_df['stkcd'].astype(str)
    for i in range(3, Factors_df.shape[1]):
        Factors_df.iloc[:, i] = pd.to_numeric(Factors_df.iloc[:, i]).astype(float)

    Index_df['Date'] = pd.to_datetime(Index_df['Date'], format='%Y%m%d')

    # Indexing by date, and create a new DataFrame as "Factors_df_date"
    Factors_df_date = Factors_df.sort_values(by='trddt')
    Factors_df_date = Factors_df_date.set_index('trddt', drop=True)
    Index_df_date = Index_df.set_index('Date', drop=True)

    # Look at which factors are missing the most
    Factors_na = (Factors_df_date.isnull().sum() / len(Factors_df_date)) * 100
    Factors_na = Factors_na.drop(Factors_na[Factors_na == 0].index).sort_values(ascending=False)
    Factors_missing_data = pd.DataFrame({'Missing Ratio': Factors_na})

    # Discard the ones whose missing ratio > 10%
    Factors_ToDrop = Factors_missing_data[Factors_missing_data['Missing Ratio'] >= 20].index.values.tolist()

    # Drop the columns that missing data the most
    New_Factor_df_date = Factors_df_date.drop(columns=Factors_ToDrop)

    # Drop rows with NAN fret
    New_Factor_df_date = New_Factor_df_date[np.isfinite(New_Factor_df_date['fret'])]

    # Fill with mean data or zero
    New_Factor_df_date = New_Factor_df_date.fillna(New_Factor_df_date.mean())
    # New_Factor_df_date = New_Factor_df_date.fillna(0)

    # There should be no null data in the Factor dataframe now.

    # Since there may be more than one stock info on each day, we extract the actual dates
    # that are recorded in Factor_df
    DaysRecorded = pd.Series(New_Factor_df_date.index.values)
    # Find unique days
    DaysRecorded = pd.Series(DaysRecorded.unique())
    # This DaysRecorded is for Index_df_date, i.e. next day return
    DaysRecorded = DaysRecorded.append(pd.Series(DaysRecorded.iloc[-1] + timedelta(days=1)), ignore_index=True).iloc[1:]
    # Pick the same date range for the Index return and forward a day after
    Idxrtn_series = Index_df_date[['idxret']]

    ### By now we will be working on the following 2 dataframes:
    # Raw data for Factor_60, sorted by stocks: Factors_df
    # Raw data for Index return, sorted by dates: Index_df
    # Raw data for Factor_60, sorted by dates: Factors_df_date
    # Matched data for Index return, filtered by the dates from Factors_df_date: Idxrtn_series

    # Drop index name
    Idxrtn_series.index.name = None
    # Filter the date according to Daysrecorded in factor_df
    Idxrtn_series = Idxrtn_series.loc[DaysRecorded]
    # check if there is any missing
    missing_idxrtn = Idxrtn_series['idxret'].index[Idxrtn_series['idxret'].apply(np.isnan)]
    # drop the missing index
    Idxrtn_series = Idxrtn_series.drop(labels=missing_idxrtn)
    # drop the same missing index - 1day in factor, since there is no data that day in Index_df
    New_Factor_df_date = New_Factor_df_date.drop(labels=(missing_idxrtn - timedelta(days=1)))

    # Create correlation matrix
    Factor_corr = New_Factor_df_date.iloc[:, 2:].corr().abs()

    # Select upper triangle of correlation matrix
    upper = Factor_corr.where(np.triu(np.ones(Factor_corr.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 1)]
    New_Factor_df_date.loc[New_Factor_df_date['fret'] > 0.1, 'fret'] = 0.1
    New_Factor_df_date.loc[New_Factor_df_date['fret'] < -0.1, 'fret'] = -0.1


    # Drop features
    New_Factor_df_date = New_Factor_df_date.drop(New_Factor_df_date[to_drop], axis=1)

    # Set column 'stkcd' column as string
    New_Factor_df_date['stkcd'] = New_Factor_df_date['stkcd'].astype(str)



    return New_Factor_df_date, Idxrtn_series