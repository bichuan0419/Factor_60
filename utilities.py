import pandas as pd
import os, math
import numpy as np
import lightgbm as lgb
from datetime import timedelta
import seaborn as sns
#Set color palette for graphs
sns.set_palette(sns.color_palette('hls', 7))
import warnings
# https://cvxopt.org/userguide/coneprog.html, https://scaron.info/blog/quadratic-programming-in-python.html
# solves 1/2 x^T P x + q^T x, subject to Gx <= h, Ax = b
import cvxopt
from cvxopt import matrix,solvers
solvers.options['show_progress'] = False
warnings.filterwarnings("ignore")


# Define a one hot encoder that return a vector of selected stocks
def get_one_hot(curr_stkid, all_stkid):
    # all stkid should be in the form of a list: ['600116','70','112',...]
    # return should be an array
    one_hot_encoded_df = pd.get_dummies(all_stkid)
    return one_hot_encoded_df[curr_stkid].T.sum().clip(0, 1).values


# Define a decoder that transforms a vector of 0 and 1's to real stock id
def inv_one_hot(stock_vec, all_stkid):
    nonzeroind = np.nonzero(stock_vec)[0]
    return all_stkid[nonzeroind]


def fetch_df_date(All_df, date=None):
    return All_df[All_df.index == date]


# Define stocks to buy given a list of factors
def Stock_to_buy(factors, curr_df, all_stkid, percentile=10):
    shares = {all_stkid[i]: 0 for i in range(len(all_stkid))}
    for factor in factors:
        temp_df = curr_df.sort_values(by=factor, ascending=False)
        df_to_choose = temp_df.head(math.ceil(len(temp_df) * (percentile / 100)))

        #         print(df_to_choose.index.unique())
        stock_to_choose = df_to_choose['stkcd'].unique()
        for stock in stock_to_choose:
            cur_value = shares.get(stock)
            shares.update({stock: cur_value + 1})

    return shares

# Define stocks to buy given a list of factors
def Stock_to_buy_DOW(factors, curr_df, all_stkid, prev_stock, dow = 4, percentile=10):
    shares = {all_stkid[i]: 0 for i in range(len(all_stkid))}
    for factor in factors:
        temp_df = curr_df.sort_values(by=factor, ascending=False)
        df_to_choose = temp_df.head(math.ceil(len(temp_df) * (percentile / 100)))

        #         print(df_to_choose.index.unique())
        stock_to_choose = df_to_choose['stkcd'].unique()
        for stock in stock_to_choose:
            cur_value = shares.get(stock)
            shares.update({stock: cur_value + 1})

        # get which weekday current day is
        currdate = curr_df.index.unique()[0]
        curr_wkday = currdate.weekday()
        if curr_wkday == dow:
            set1 = set(prev_stock.items())  # previous stocks in hold
            set2 = set(shares.items())  # current stocks to buy
            diff_keys = [a[0] for a in (set1 - set2)]  # names of the stocks that needs to calculate
            for key in diff_keys:
                # get difference in number of shares to trade
                difference = shares.get(key) - prev_stock.get(key)
                if difference > 0:
                    # only sell is allowed
                    shares.update({key: 0})
    return shares


def custom_metric(y_true, y_pred):
    residual = abs(y_true - y_pred).astype("float")
    loss = residual**5
    return "custom_metric", np.mean(loss), False

def Stock_to_buy_LGBM(curr_df, all_stkid,  percentile=10):
    lgbm = lgb.LGBMRegressor()
    days = curr_df.index.unique()
    startdate = days[0]
    enddate = days[-2]
    testdate = days[-1]

    train_df = filter_df_by_date(startdate=startdate, enddate=enddate, df=curr_df)
    test_df = filter_df_by_date(startdate=testdate, enddate=testdate, df=curr_df)

    X_test = test_df.iloc[:, 2:]
    y_test = test_df.iloc[:, [0, 1]]

    lgbm.fit(train_df.iloc[:, 2:], train_df.iloc[:, 1],eval_metric=custom_metric)
    y_pred = lgbm.predict(X_test)

    y_test['pred'] = y_pred
    temp_df = y_test.sort_values(by='pred', ascending=False)
    df_to_choose = temp_df.head(math.ceil(len(temp_df) * (percentile / 100)))

    #         print(df_to_choose.index.unique())
    stock_to_choose = df_to_choose['stkcd'].unique()
    shares = {all_stkid[i]: 0 for i in range(len(all_stkid))}
    for stock in stock_to_choose:
        cur_value = shares.get(stock)
        shares.update({stock: cur_value + 1})
    return shares

def vote_for_stocks(df,testdate,all_stkid,percentile):
    curr_df = filter_df_by_date(df, testdate, testdate)
    factors = curr_df.columns[2:].tolist()
    shares_vote = {all_stkid[i]: 0 for i in range(len(all_stkid))}

    for factor in factors:
        curr_df_sort_temp = curr_df.sort_values(by=factor,ascending=False)
        df_to_choose = curr_df_sort_temp.head(math.ceil(len(curr_df_sort_temp) * (percentile / 100)))
        stock_to_choose = df_to_choose['stkcd'].unique()
        for stock in stock_to_choose:
            cur_value = shares_vote.get(stock)
            shares_vote.update({stock: cur_value + 1})

    sorted_score = sorted(shares_vote.items(), key=lambda kv: kv[1], reverse=True)
    # shares = {all_stkid[i]: 0 for i in range(len(all_stkid))}
    # for i in sorted_score:
    #     if i[1] > 20:
    #         shares.update(dict([i]))
    return dict(sorted_score)

def vote_for_stocks_DOW(df,testdate,all_stkid, prev_stock, dow = 4, percentile=10):
    curr_df = filter_df_by_date(df, testdate, testdate)
    factors = curr_df.columns[2:].tolist()
    shares_vote = {all_stkid[i]: 0 for i in range(len(all_stkid))}

    for factor in factors:
        curr_df_sort_temp = curr_df.sort_values(by=factor,ascending=False)
        df_to_choose = curr_df_sort_temp.head(math.ceil(len(curr_df_sort_temp) * (percentile / 100)))
        stock_to_choose = df_to_choose['stkcd'].unique()
        for stock in stock_to_choose:
            cur_value = shares_vote.get(stock)
            shares_vote.update({stock: cur_value + 1})

    sorted_score = sorted(shares_vote.items(), key=lambda kv: kv[1], reverse=True)
    shares = dict(sorted_score)
    # get which weekday current day is
    curr_wkday = testdate.weekday()
    if curr_wkday == dow:
        set1 = set(prev_stock.items())  # previous stocks in hold
        set2 = set(shares.items())  # current stocks to buy
        diff_keys = [a[0] for a in (set1 - set2)]  # names of the stocks that needs to calculate
        for key in diff_keys:
            # get difference in number of shares to trade
            difference = shares.get(key) - prev_stock.get(key)
            if difference > 0:
                # only sell is allowed
                shares.update({key: 0})
    # for i in sorted_score:
    #     if i[1] > 20:
    #         shares.update(dict([i]))
    return shares


def calc_commission_fee(stock_to_buy, prev_stock=None, commission_fee=0.0015):
    # all stock are dictionaries with stock id as key and size as value
    total_commission = 0

    set1 = set(prev_stock.items()) # previous stocks in hold
    set2 = set(stock_to_buy.items()) # current stocks to buy
    diff_keys = [a[0] for a in (set1 - set2)] # names of the stocks that needs to calculate

    # total_shares = 1
    for key in diff_keys:
        # get difference in number of shares to trade
        a = stock_to_buy.get(key)
        b = prev_stock.get(key)
        difference =  a-b
        # if difference < 0: # NEED TO BUY THIS STOCK
        total_commission += abs(difference) * commission_fee

            # else: # the stocks need to be sold
            #     total_commission += prev_stock.get(key) * commission_fee


    return total_commission


def filter_df_by_date(df, startdate, enddate):
    mask = ((startdate <= df.index) & (enddate >= df.index))
    return df[mask]


def max_drawdown(return_hist):
    # we want max_drawdown perday
    return np.min([(return_hist[i + 1] - return_hist[i])/return_hist[i] for i in range(len(return_hist)-1)])


def CalcReturns(stock_to_buy, currdate, New_Factor_df_date, Idxrtn_series, all_stkid, prev_stock=None, stocks_chosen=None):
    # factors are labeled as ['tot_rank1', 'tot_rank2']...
    # df are the dataframe that includes returns, stocks and factors at the date specified
    # prev_stock are labeled as a dictionary as {'stkid1':0, 'stkid2':5,...}, where the keys are sizes of stock in hold

    # initialize
    returns = 0
    # get dataframe for current date
    curr_df = fetch_df_date(New_Factor_df_date, date=currdate)

    # get returns for the next day
    # next day is not the actual next day from calendar, but the next day index from our factor dataframe
    nextday_index = New_Factor_df_date.index.unique().get_loc(currdate)
    nextday = Idxrtn_series.index[nextday_index]
    #     print(nextday)
    # it should return 'somedate' 'somevalue'
    curr_idx_ret = fetch_df_date(Idxrtn_series, date=nextday)

    # Calculate commission fee
    commission_fee = calc_commission_fee(stock_to_buy, prev_stock=prev_stock, commission_fee=0.0015)

    # # total shares (need to divide total return by this amount to get an averaged value)
    total_shares = sum(stock_to_buy.values())

    # Find today's return
    for key,value in stock_to_buy.items():
        # locate the stocks that are not labeled as 0
        if stock_to_buy.get(key) != 0:
            # add today's return and substract next days return
            returns += curr_df.loc[curr_df['stkcd'] == key, 'fret'].iloc[0]*value

    # Subtract commission fee
    returns -= commission_fee
    #     print('commission fee:', commission_fee)
    # if total_shares != 0:
    if total_shares != 0:
        returns /= total_shares
    else:
        returns = 0

    returns -= curr_idx_ret.values[0][0]
    return returns

def convert_stock_to_dict(stock_names, all_stkid):
    temp_stock = get_one_hot(stock_names,all_stkid)
    stock_to_buy = {all_stkid[j]:temp_stock[j] for j in range(len(all_stkid))}
    return stock_to_buy


# def pick_factor_within_period(startdate, enddate, df, New_Factor_df_date, Idxrtn_series, all_stkid, number_factors=3):
#     # df is a segment, or the complete dataframe containing returns and factors
#     factors = df.columns.tolist()[2:]
#     period_df = filter_df_by_date(df, startdate, enddate)
#     datelist = period_df.index.unique()
#
#     # Initialize a dictionary counting # of appearance of factors that were the best
#     factor_scores = {factors[i]: 0 for i in range(len(factors))}
#
#     for factor in factors:
#         # reset previous stocks for each new factor
#         prev_stock = {all_stkid[i]: 0 for i in range(len(all_stkid))}
#         # reset tempoary total for that factor
#         temp_total = 1
#         # reset Total return to record the history of temp_total
#         Total_Return = []
#         # reset RoMaD
#         temp_profit = 0
#         temp_max_drawdown = 0
#         RoMaD = 0
#
#         # This loop calculate the profit history for a single factor within a period of time
#         for day in datelist:
#             # curr_df contains in a specific day, all stock returns together with a specific factor
#             curr_df = period_df[period_df.index == day][['fret', 'stkcd', factor]]
#
#             # choose that stock, as a dictionary
#             stock_to_buy = Stock_to_buy([factor], curr_df, all_stkid, percentile=10)
#             temp_total *= (1+CalcReturns(stock_to_buy, currdate=day, New_Factor_df_date = New_Factor_df_date, Idxrtn_series = Idxrtn_series, all_stkid = all_stkid, prev_stock=prev_stock))
#
#             Total_Return.append(temp_total)
#             #             print(Total_Return)
#             prev_stock = stock_to_buy
#
#         # Now calculate analyzers for the portfolio created according to the specific factor
#         # profit:
#
#         temp_profit = Total_Return[-1] - Total_Return[0]
#         # print(temp_profit)
#         #         plt.plot(datelist,Total_Return)
#         # max_drawdown is the smallest number (very likely to be a negative number) over a period of time
#         temp_max_drawdown = max_drawdown(Total_Return)
#         # define a criterion we want to use (May subject to change!)
#         RoMaD = temp_profit / temp_max_drawdown
#         # factor_scores.update({factor: RoMaD})
#         factor_scores.update({factor: temp_max_drawdown})
#
#     sorted_score = sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=True)
#
#     return [i[0] for i in sorted_score[0:number_factors]]

def pick_factor_within_period_DoW(startdate, enddate, df, New_Factor_df_date, Idxrtn_series, all_stkid, number_factors=3):
    # df is a segment, or the complete dataframe containing returns and factors
    factors = df.columns.tolist()[2:]
    period_df = filter_df_by_date(df, startdate, enddate)
    datelist = period_df.index.unique()

    # Initialize a dictionary counting # of appearance of factors that were the best
    factor_scores = {factors[i]: 0 for i in range(len(factors))}

    for factor in factors:
        # reset previous stocks for each new factor
        prev_stock ={all_stkid[i]: 0 for i in range(len(all_stkid))}
        # reset tempoary total for that factor
        temp_total = 1
        # reset Total return to record the history of temp_total
        Total_Return = []
        # reset RoMaD
        temp_profit = 0
        temp_max_drawdown = 0
        RoMaD = 0

        # This loop calculate the profit history for a single factor within a period of time
        for day in datelist:
            curr_df = df[df.index == day]

            # Stock_to_buy(factors=selected_features, curr_df=curr_df, all_stkid=all_stkid, percentile=10)
            stock_to_buy = Stock_to_buy_DOW(factors=[factor], curr_df=curr_df, all_stkid=all_stkid,
                                            prev_stock=prev_stock, dow=4, percentile=10)

            temp_total *= (1+CalcReturns(stock_to_buy, currdate=day, New_Factor_df_date = New_Factor_df_date, Idxrtn_series = Idxrtn_series, all_stkid = all_stkid, prev_stock=prev_stock))

            Total_Return.append(temp_total)
                #             print(Total_Return)
            prev_stock = stock_to_buy

        # Now calculate analyzers for the portfolio created according to the specific factor
        # profit:

        temp_profit = Total_Return[-1] - Total_Return[0]
        # print(temp_profit)
        #         plt.plot(datelist,Total_Return)
        temp_max_drawdown = max_drawdown(Total_Return)
        # define a criterion we want to use (May subject to change!)
        RoMaD = temp_profit / temp_max_drawdown
        # factor_scores.update({factor: RoMaD})
        factor_scores.update({factor: RoMaD})

    sorted_score = sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=True)

    return [i[0] for i in sorted_score[0:number_factors]]

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))
#
def pick_factor_within_period_QuadProg(startdate, enddate, df):
    currFolder = os.getcwd()
    period_df = filter_df_by_date(df, startdate, enddate)
    factors = period_df.columns[2:].tolist()
    datelist = period_df.index.unique()

    m = len(factors)
    n = len(datelist)
    Total_Return_Rate = np.zeros((m, n))
    for factor in factors:
        index_factor = factors.index(factor)
        string_path = 'log_data/' + factor + '_log_file.csv'
        temp_path = os.path.join(currFolder, string_path)
        # create temporary dataframe to extract data
        col_names = ['trddt', 'return_rate']
        log_file = pd.read_csv(temp_path, index_col=0, names=col_names)
        log_file.index = pd.to_datetime(log_file.index)
        # get date required
        temp_return_rate = filter_df_by_date(startdate=startdate, enddate=enddate, df=log_file)
        Total_Return_Rate[index_factor, :] = temp_return_rate.values.reshape((1, len(temp_return_rate)))

    # strategy mean
    strategy_mean = np.mean(Total_Return_Rate, axis=1)
    # strategy_covariance
    strategy_cov = np.cov(Total_Return_Rate)
    # strategy_variance
    strategy_var = np.diag(strategy_cov)
    #     print(strategy_var)
    P = (strategy_cov + (10 ** (-15)) * np.eye(len(strategy_cov))).astype('double')
    q = np.zeros((m, 1))
    G = -np.concatenate((np.eye(m), strategy_mean.reshape(1, m)))
    h = np.zeros((m + 1, 1))
    # A = np.ones((1, m))
    A = None
    b = None
    # b = np.array([20.])
    x = cvxopt_solve_qp(P, q, G=G, h=h, A=A, b=b)
    largest_index = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:1]

    return [factors[i] for i in largest_index]


def convert_return_rate_to_market_cap(return_rate_df):
    datelist = return_rate_df.index.tolist()
    temp_total = 1
    Total_Return = []
    for day in datelist:
        temp_total *= (1 + return_rate_df.loc[day, 'return_rate'])
        Total_Return.append(temp_total)

    return Total_Return


def pick_factor_within_period(startdate, enddate, New_Factor_df_date, number_factors=3):
    currFolder = os.getcwd()
    factors = New_Factor_df_date.columns[2:].tolist()
    # Initialize a dictionary counting # of appearance of factors that were the best
    factor_scores = {factors[i]: 0 for i in range(len(factors))}

    for factor in factors:
        string_path = 'log_data/' + factor + '_log_file.csv'
        temp_path = os.path.join(currFolder, string_path)
        # create temporary dataframe to extract data
        col_names = ['trddt', 'return_rate']
        log_file = pd.read_csv(temp_path, index_col=0, names=col_names)
        log_file.index = pd.to_datetime(log_file.index)
        # get date required
        temp_return_rate = filter_df_by_date(startdate=startdate, enddate=enddate, df=log_file)

        # reset Total return to record the history of temp_total
        Total_Return = convert_return_rate_to_market_cap(temp_return_rate)

        temp_max_drawdown = max_drawdown(Total_Return)

        factor_scores.update({factor: temp_max_drawdown})

    sorted_score = sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=False)

    return [i[0] for i in sorted_score[0:number_factors]]


# def reversed_sort(yesterday, today, df, number_factors=3):
#     yesterday_fret = df[df.index == yesterday].loc[:, ['stkcd', 'fret']]
#     today_factors = df[df.index == today].iloc[:,
#                     [i for i in range(len(df.columns)) if i != 1]]
#     result = pd.merge(yesterday_fret, today_factors, on='stkcd').dropna()
#
#     temp_df = result.sort_values(by='fret', ascending=False)
#     df_to_choose = temp_df.head(math.ceil(len(temp_df) * (10 / 100)))
#     mean_df = df_to_choose.iloc[:, 2:].mean()
#     largest_index = sorted(range(len(mean_df)), key=lambda i: mean_df[i], reverse=True)[:number_factors]
#
#     return mean_df[largest_index].index.tolist()
