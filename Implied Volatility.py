# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:16:09 2021

@author: user3
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


def conbond_option_price(df_conbond):
    df_Sv = obtain_data('strbvalue')
    convratio = obtain_data('convratio')
    for col in convratio.columns:
        sub_convratio = convratio[col].dropna()
        if len(sub_convratio) != 0:
            start_convratio = sub_convratio.index[0]
            end_convratio = sub_convratio.index[-1]
            spec_index = convratio.loc[start_convratio:end_convratio, col][
                convratio.loc[start_convratio:end_convratio, col] <= 0].index
            convratio.loc[spec_index, col] = np.nan
            convratio.loc[start_convratio:end_convratio, col].fillna(method='ffill', inplace=True)
            df_Sv.loc[start_convratio:end_convratio, col].fillna(method='ffill', inplace=True)
    df_Op = df_conbond - df_Sv
    return df_Op / convratio


def maturity_day(df_conbond):
    '''read link data can obtain from .mat, but the xlsx including the start day and end day of the convbond data, 需要债券到期日之类的数据来计算隐含波动率 '''
    a = obtain_data('B_WINDCODE_Linked', dtype='str', level1='basicinfo')
    #     df_date=pd.read_csv('TradeDate.csv',index_col='S_INFO_WINDCODE',parse_dates=['B_INFO_CARRYDATE','B_INFO_ENDDATE'])
    a.insert(1, 'Option_Year', obtain_data('B_INFO_TERM_YEAR_', level1='basicinfo')['B_INFO_TERM_YEAR_'])
    a.insert(1, 'Option_Day', obtain_data('B_INFO_TERM_DAY_', level1='basicinfo')['B_INFO_TERM_DAY_'])
    a.insert(1, 'End_Date', read_date('B_INFO_ENDDATE', level1='basicinfo'))
    a.insert(1, 'Maturity', read_date('B_INFO_ENDDATE', level1='basicinfo'))
    a.insert(1, 'Start_Date', read_date('B_INFO_CARRYDATE', level1='basicinfo'))
    a = a.loc[a['Start_Date'].dropna().index, :]

    for i in df_conbond.columns:
        if len(df_conbond[i].dropna()) != 0:
            a.loc[i, 'End_Date'] = df_conbond[i].dropna().index[-1]
        else:
            if i in a.index:
                a = a.drop(labels=i, axis=0)
    #     a=a.rename(columns={'1':'Index_Stock'})
    a = a.loc[a['Index_Stock'].dropna().index, :]
    a = a.loc[a['Index_Stock'] != '\x00\x00', :]
    return a


def bsm_call_value(s, k, T, r, sigma):
    s = float(s)
    d1 = (log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sqrt(T) * sigma)
    d2 = (log(s / k) + (r - 0.5 * sigma ** 2) * T) / (sqrt(T) * sigma)
    value = (s * stats.norm.cdf(d1, 0.0, 1.0)) - k * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)
    return value


def sigma(C, S, K, T, r):
    if C <= bsm_call_value(S, K, T, r, 0.0000001):
        return 0
    elif bsm_call_value(S, K, T, r, 1) <= C:
        return 1
    r_sigma = 1
    l_sigma = 0
    mid_sigma = (r_sigma + l_sigma) / 2

    threshold = 0.00001
    bs_P = bsm_call_value(S, K, T, r, mid_sigma)
    while (abs(bs_P - C) > threshold):
        if bs_P - C > 0:
            r_sigma = mid_sigma
            mid_sigma = (mid_sigma + l_sigma) / 2
        else:
            l_sigma = mid_sigma
            mid_sigma = (mid_sigma + r_sigma) / 2
        bs_P = bsm_call_value(S, K, T, r, mid_sigma)
    return mid_sigma


def sigma_matirx(df_conbond, df_stock, df_T, start_date, end_date):
    df_sigma = pd.DataFrame()
    df_C = conbond_option_price(df_conbond)
    df_S = df_stock

    convratio = obtain_data('convratio')
    for col in convratio.columns:
        sub_convratio = convratio[col].dropna()
        if len(sub_convratio) != 0:
            start_convratio = sub_convratio.index[0]
            end_convratio = sub_convratio.index[-1]
            spec_index = convratio.loc[start_convratio:end_convratio, col][
                convratio.loc[start_convratio:end_convratio, col] <= 0].index
            convratio.loc[spec_index, col] = np.nan
            convratio.loc[start_convratio:end_convratio, col].fillna(method='ffill', inplace=True)

    #     df_K = obtain_data('convprice')
    df_K = df_conbond / convratio
    for i in tqdm(df_conbond.loc[start_date:end_date].index):
        tradable_list = set(df_conbond.loc[i, :].dropna().index).intersection(df_T.index)
        for j in tradable_list:
            C = df_C.loc[i, j]
            S = df_S.loc[i, df_T.loc[j, 'Index_Stock']]
            K = df_K.loc[i, j]
            T = int(str(df_T.loc[j, 'Maturity'] - i).split(' ')[0]) / df_T.loc[j, 'Option_Day'] * df_T.loc[
                j, 'Option_Year']
            #             r = 0.1
            r = 0.01
            if ~any(np.isnan([C, S, K])) and T > 0:
                df_sigma.loc[i, j] = sigma(C, S, K, T, r)
    return df_sigma




if __name__ == "__main__":
    start_day = '2004-01-02'
    end_day = '2020-12-28'
    df_conbond = obtain_data('originclose')

    df_stock = stock_close()
    df_T = maturity_day(df_conbond)
    df_sigma = sigma_matirx(df_conbond, df_stock, df_T, start_day, end_day)
