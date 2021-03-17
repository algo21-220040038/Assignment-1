# -*- coding: utf-8 -*-
"""
Created on Mon March 01 16:16:09 2021

@author: YvesYang
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from math import log, sqrt, exp
from scipy import stats
from tqdm import tqdm, trange
import matplotlib.ticker as ticker
import datetime

alpha_path = './杨阳工作文件/可转债/'
conv_path = './杨阳工作文件/可转债/'


def num_to_date(num):
    num = datetime.date.fromordinal(int(num) - 366)
    num = num.strftime('%Y-%m-%d')
    return num


def read_str_alpha(level2='stock_number_wind', level1='basicinfo'):
    data = []
    with h5py.File(alpha_path + "alpha.mat", 'r') as f:
        for column in f[level1][level2]:
            row_data = []
            for row_number in range(len(column)):
                row_data.append(''.join(map(chr, f[column[row_number]][:])))
            data.append(row_data)
    return np.transpose(data)


def read_int_alpha(level2='close', level1='dailyinfo'):
    data = []
    with h5py.File(alpha_path + "alpha.mat", 'r') as f:
        for column in f[level1][level2]:
            data.append(column)
    return np.transpose(data)


def read_date_alpha(level2='dates', level1='dailyinfo'):
    data = []
    with h5py.File(alpha_path + "alpha.mat", 'r') as f:
        for column in f[level1][level2]:
            data.append(column)
    data = np.transpose(data)
    data = map(num_to_date, data.squeeze())
    return list(data)


# 通过读取的文件转换为dataframe
def stock_close(close):
    a = read_date_alpha()
    b = read_str_alpha('stock_number_wind')
    data = read_int_alpha(close)
    data = pd.DataFrame(data, index=b.squeeze())
    data = data.T.set_index(np.array(pd.to_datetime(a)))
    return data


def read_str(level2='B_WINDCODE', level1='basicinfo', source_file=conv_path + "./Convbond.mat"):
    data = []
    with h5py.File(source_file, 'r') as f:
        for column in f[level1][level2]:
            row_data = []
            for row_number in range(len(column)):
                row_data.append(''.join(map(chr, f[column[row_number]][:])))
            data.append(row_data)
    return np.transpose(data).squeeze()


def read_date(level2='dates_8n', level1='dailyinfo', source_file=conv_path + "./Convbond.mat"):
    data = []
    with h5py.File(source_file, 'r') as f:
        for column in f[level1][level2]:
            data.append(column)
    data = np.transpose(data)
    data = np.array([str(i)[:8] for i in data.squeeze()])
    data = pd.to_datetime(data)
    return data


def obtain_data(level2, dtype='int', level1='dailyinfo', source_file=conv_path + "./Convbond.mat"):
    if level1 == 'dailyinfo':
        # 获取交易日期
        date = read_date()
        # 获取转债index(读取string)
        index = read_str()

        # 获取转债交易数据
        data = []
        with h5py.File(source_file, 'r') as f:
            for column in f[level1][level2]:
                data.append(column)
        data = np.transpose(data)

        # 添加日期index和股票对应代码
        data = pd.DataFrame(data, index=pd.Series(index, name='Index_Bond'))
        data = data.T.set_index(date)
        # 筛选convbond
        bond = pd.DataFrame(pd.Series(read_str('CB_LIST_TYPE'), name='CB_LIST_TYPE'))
        bond = bond.set_index(index)
        choosed_list = ['优先配售,网上定价和网下配售', '优先配售和上网定价', '优先配售，网下配售', '网下发行']
        index = []
        for i in choosed_list:
            index.extend(list(bond[bond['CB_LIST_TYPE'] == i].index))
        data = data.loc[:, pd.Series(index)]

    elif level1 == 'basicinfo' and dtype == 'str':
        data = []
        with h5py.File(source_file, 'r') as f:
            for column in f[level1][level2]:
                row_data = []
                for row_number in range(len(column)):
                    row_data.append(''.join(map(chr, f[column[row_number]][:])))
                data.append(row_data)
                # 获取转债index(读取string)
        index = read_str()
        # 添加日期index和股票对应代码
        data = pd.DataFrame(data[1], index=pd.Series(index, name='Index_Bond'))

        data = data.rename(columns={0: 'Index_Stock'})
    #         data=data.loc[data['Index_Stock']!='\x00\x00',:]

    elif level1 == 'basicinfo' and dtype == 'int':
        data = []
        with h5py.File(source_file, 'r') as f:
            for column in f[level1][level2]:
                data.append(column)
        # 获取转债index(读取string)
        index = read_str()
        # 添加日期index和股票对应代码
        data = pd.DataFrame(data[0], index=pd.Series(index, name='Index_Bond'))
        data = data.rename(columns={0: level2})
    return data




# 返回债券余额
def bond_amount(df_conbond):
    df_among_change = pd.read_csv('convbond_amount.csv')
    index = df_conbond.columns
    date = df_conbond.index

    df_try = pd.DataFrame(data=None, columns=index, index=date)
    for i in index:
        change_day = df_among_change['S_INFO_CHANGEDATE'][df_among_change['S_INFO_WINDCODE'] == i]
        if len(change_day) >= 1:
            change_day = sorted(change_day, reverse=False)
            change_day = [pd.to_datetime(str(i)) for i in change_day]
            change_amount = df_among_change['B_INFO_OUTSTANDINGBALANCE'][df_among_change['S_INFO_WINDCODE'] == i]
            change_amount = sorted(change_amount, reverse=True)
            for j in range(len(change_day) - 1):
                start_day = change_day[j]
                end_day = change_day[j + 1]
                df_try.loc[start_day:end_day, i] = change_amount[j]
            if len(change_day) == 1:
                df_try.loc[change_day[0]:, i] = change_amount[0]
            elif len(df_try[i].dropna()) != 0:
                final_day = df_conbond[i].dropna().index[-1] if len(df_conbond[i].dropna()) != 0 else \
                df_conbond[i].index[-1]
                df_try.loc[end_day:end_day, i] = change_amount[-1]
                df_try.loc[end_day:final_day, i] = df_try[i].dropna()[-1]
    return df_try


def bond_hold(day, lpercent, hpercent, df_T, df_conbond, df_bond_amount, df_sigma, df_stock_sigma):
    df_Hold = pd.DataFrame(data=None, columns=['bonprice', 'amount', 'share', 'bond_sigma', 'end_day'])
    bond_index = df_sigma.loc[day, :].dropna().index
    stock_index = df_T.loc[bond_index]['Index_Stock']
    diff_sigma = np.array(df_sigma.loc[day, bond_index]) - np.array(df_stock_sigma.loc[day, stock_index])
    hold_list = bond_index[np.argsort(diff_sigma)][int(len(bond_index) * lpercent): int(len(bond_index) * hpercent)]
    # reverse position
    df_Hold.loc[:, 'bonprice'] = df_conbond.loc[day, hold_list]
    df_Hold.loc[:, 'amount'] = df_bond_amount.loc[day, hold_list]
    df_Hold.loc[:, 'share'] = df_bond_amount.loc[day, hold_list] / sum(df_bond_amount.loc[day, hold_list].dropna())
    df_Hold.loc[:, 'bond_sigma'] = df_sigma.loc[day, hold_list]
    df_Hold.loc[:, 'end_day'] = df_T.loc[hold_list, 'End_Date']
    #     df_Hold=df_Hold.replace(0,np.nan)
    df_Hold = df_Hold.dropna()
    return df_Hold


def bench_bond(day, df_T, df_conbond, df_bond_amount, df_sigma, df_stock_sigma):
    df_Hold = pd.DataFrame(data=None, columns=['bonprice', 'amount', 'share', 'bond_sigma', 'end_day'])
    bond_index = df_sigma.loc[day, :].dropna().index
    hold_list = bond_index
    df_Hold.loc[:, 'bonprice'] = df_conbond.loc[day, hold_list]
    df_Hold.loc[:, 'amount'] = df_bond_amount.loc[day, hold_list]
    df_Hold.loc[:, 'share'] = df_bond_amount.loc[day, hold_list] / sum(df_bond_amount.loc[day, hold_list].dropna())
    df_Hold.loc[:, 'bond_sigma'] = df_sigma.loc[day, hold_list]
    df_Hold.loc[:, 'end_day'] = df_T.loc[hold_list, 'End_Date']
    #     df_Hold=df_Hold.replace(0,np.nan)
    df_Hold = df_Hold.dropna()
    return df_Hold


# 填充Nan值
def modified_bond(df_conbond):
    df_conbond = obtain_data('originclose')
    for c in df_conbond.columns:
        if len(df_conbond[c].dropna().index) != 0:
            s_day = df_conbond[c].dropna().index[0]
            e_day = df_conbond[c].dropna().index[-1]
            if len(df_conbond.loc[:e_day]) + 100 <= 5573:
                append_day = df_conbond.index[len(df_conbond.loc[:e_day]) + 100]
            else:
                append_day = df_conbond.index[5573]
            df_conbond.loc[e_day:append_day, c] = df_conbond.loc[e_day, c]
            for idx in np.argwhere(np.array(np.isnan(df_conbond.loc[s_day:e_day, c]))):
                df_conbond.loc[df_conbond.index[idx[0] + len(df_conbond.loc[:s_day]) - 1], c] = df_conbond.loc[
                    df_conbond.index[idx[0] + len(df_conbond.loc[:s_day]) - 2], c]
    return df_conbond


def daily_result(Return_list, df_weight, df_cost, df_conbond):
    sub_return = []
    #     sub_return_rate=[]
    for i in range(len(df_weight) - 1):
        sub_return.append(Return_list[i])
        Return_start = sub_return[-1]
        day = df_weight.index[i]
        day_next = df_weight.index[i + 1]
        for j in df_conbond.loc[day:day_next].index[1:-1]:
            Return = sum((df_conbond.loc[j] / df_cost.loc[day] * df_weight.loc[day]).dropna()) * Return_start
            sub_return.append(Return)
    #             sub_return_rate.append(Return/Return_start)
    return sub_return


def backtest(lpercent, hpercent, cost, df_sigma, start_day='2004-01-02', end_day='2020-06', period=5, Bench=False):
    # 需要的数据
    df_conbond = obtain_data('originclose')
    df_stock_adj = stock_close('close_adj')
    df_T = maturity_day(df_conbond)
    df_bond_amount = bond_amount(df_conbond)
    df_stock_sigma = (df_stock_adj.loc[:end_day, :] / df_stock_adj.loc[:end_day, :].shift(1)).rolling(
        60).std() * np.sqrt(250)

    Return_list = [1]
    Return = 1
    Not_trade = pd.DataFrame()
    df_weight = pd.DataFrame(data=0, index=df_sigma[start_day:end_day:period].index, columns=df_conbond.columns)
    df_cost = pd.DataFrame(data=0, index=df_sigma[start_day:end_day:period].index, columns=df_conbond.columns)
    # 按period回测
    for i in range(len(df_sigma[start_day:end_day:period].index) - 1):
        # 确定交易日期
        day = df_sigma[start_day:end_day:period].index[i]
        day_next = df_sigma[start_day:end_day:period].index[i + 1]
        # 获取持仓
        if Bench:
            df_hold = bench_bond(day, df_T, df_conbond, df_bond_amount, df_sigma, df_stock_sigma)
        else:
            df_hold = bond_hold(day, lpercent, hpercent, df_T, df_conbond, df_bond_amount, df_sigma, df_stock_sigma)
        # 未交易的股票以原来的份额和买入价加入持仓
        for temp_in in Not_trade.index:
            if temp_in not in df_hold.index:
                df_hold = df_hold.append(Not_trade.loc[temp_in, :])
            else:
                df_hold.loc[temp_in, :] = Not_trade.loc[temp_in, :]
        # 仓位再平衡
        if len(Not_trade) != 0:
            diff_list = list(set(df_hold.index) - set(Not_trade.index))
            df_hold.loc[diff_list, 'share'] = (1 - Not_trade['share'].sum()) * df_hold.loc[
                diff_list, 'amount'] / df_hold.loc[diff_list, 'amount'].sum()
        # 记录持仓
        df_weight.loc[day, df_hold.index] = df_hold.loc[:, 'share']
        df_cost.loc[day, df_hold.index] = df_hold.loc[:, 'bonprice']
        # 当月提前交易的转债
        early_hold = df_hold[df_hold['end_day'] <= day]
        # 收益计算
        if len(early_hold) != 0:
            early_price_next = [df_conbond.loc[d, s] for d, s in zip(early_hold['end_day'], early_hold.index)]
            df_hold = df_hold[df_hold['end_day'] > day]
            price_next = df_conbond.loc[day_next, df_hold.index]
            nan_bool = np.isnan(price_next)
            Return = (sum(
                price_next[~nan_bool] / df_hold['bonprice'][~nan_bool] * df_hold['share'][~nan_bool] * Return) +
                      sum(early_price_next / early_hold['bonprice'] * early_hold['share'] * Return) +
                      sum(1 * df_hold['share'][nan_bool]) * Return)

            Return_list.append(Return)
            Not_trade = df_hold[nan_bool]
        else:
            price_next = df_conbond.loc[day_next, df_hold.index]
            nan_bool = np.isnan(price_next)

            Return = (sum(price_next[~nan_bool] / df_hold['bonprice'][~nan_bool] * df_hold['share'][~nan_bool] * Return)
                      + sum(1 * df_hold['share'][nan_bool] * Return))

            Return_list.append(Return)
            Not_trade = df_hold[nan_bool]

    # 将最后
    last_day_price = np.array([df_conbond.loc[d, s] for d, s in zip(Not_trade['end_day'], Not_trade.index)])
    Return_list[-1] = Return_list[-1] + sum(last_day_price / Not_trade['bonprice'] * Not_trade['share']) * Return_list[
        -2]

    # 计算交易成本
    return_rate = Return_list[1:] / np.array(Return_list[:-1])
    for i in range(len(df_weight) - 1):
        Return_list[i + 1] = return_rate[i] * (1 - sum(abs(df_weight.iloc[i, :] - df_weight.iloc[i + 1, :])) * cost) * \
                             Return_list[i]
    return Return_list, df_weight, df_cost


def picture_return(start_day, Return_weight, df_conbond, Bench_list, Return_return_Result, layer):
    end_day = Return_weight.index[-1]
    date = df_conbond.loc[start_day:end_day].index[1:]
    plt.figure(figsize=(15, 6))
    plt.plot(date, Bench_list, label='benchmark')

    for i in range(layer):
        Return_list = Return_return_Result[i]
        plt.plot(date, Return_list, label='portfolio' + str(i + 1))
        plt.plot(date, np.array(Return_list) / np.array(Bench_list), label='Extra-return' + str(i + 1))
    plt.xticks(rotation=70)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.title('The return of position')
    plt.legend()
    plt.show()


def picture_position(Bench_weight, Return_weight_Result, layer):
    plt.figure(figsize=(15, 6))
    plt.title('The number of position')
    plt.plot(Bench_weight.index[:-1], np.sum(Bench_weight != 0, axis=1)[:-1], '-d', label='benchmark')

    for i in range(layer):
        Return_weight = Return_weight_Result[i]
        plt.plot(Bench_weight.index[:-1], np.sum(Return_weight != 0, axis=1)[:-1], '-*', label='portfolio' + str(i + 1))
    plt.xticks(rotation=70)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.legend()
    plt.show()


def class_return(df_sigma, start_day, end_day, layer, period):
    df_conbond = obtain_data('originclose')
    Bench_list, Bench_weight, Bench_cost = backtest(0, 0.5, cost=0.001, df_sigma=df_sigma,
                                                    start_day=start_day, end_day=end_day, period=period, Bench=True)
    df_conbond = modified_bond(df_conbond)
    Bench_list = daily_result(Bench_list, Bench_weight, Bench_cost, df_conbond)

    class_list = [i / layer for i in range(layer + 1)]

    Return_return_Result = []
    Return_weight_Result = []
    for i in range(layer):
        lpercent = class_list[i]
        hpercent = class_list[i + 1]
        Return_list, Return_weight, Return_cost = backtest(lpercent, hpercent, cost=0.001, df_sigma=df_sigma,
                                                           start_day=start_day, end_day=end_day, period=period)
        Return_list = daily_result(Return_list, Return_weight, Return_cost, df_conbond)

        Return_return_Result.append(Return_list)
        Return_weight_Result.append(Return_weight)

    picture_return(start_day, Return_weight, df_conbond, Bench_list, Return_return_Result, layer)
    picture_position(Bench_weight, Return_weight_Result, layer)
    return Return_return_Result, Return_weight_Result, Bench_list, Bench_weight



if __name__ == "__main__":
    start_day = '2016-01-01'
    end_day = '2021-02-10'

    df_sigma = pd.read_csv('stock_sigma.csv')
    Return_return_Result,Return_weight,Bench_list,Bench_weight=class_return(df_sigma, start_day, end_day, 3 ,20)
    Return_return_Result10, Return_weight10, Bench_list10, Bench_weight10 = class_return(df_sigma, start_day, end_day,
                                                                                         3, 10)
