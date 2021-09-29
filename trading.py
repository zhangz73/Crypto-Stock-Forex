import matplotlib
matplotlib.use('Agg')
import sys, time, math, os.path
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import random
from itertools import cycle
import collections
from collections import Counter, deque
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Flatten
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
crypto_info_fname = sys.argv[1]
crypto_info = pd.read_csv(crypto_info_fname)
crypto_info_copy = crypto_info.copy()

def single_attr_stat(data):
    #return [data.mean(), data.std(), data.median(), data.iloc[-1], data.iloc[-1] - data.iloc[0]]
    return [np.mean(data), np.std(data), np.median(data), data[-1], data[-1] - data[0]]

def get_linreg(data):
    x = np.array(range(len(data))).reshape(len(data), 1)
    reg = LinearRegression().fit(x, data)
    coef = reg.coef_[0]
    loss = np.sum(np.square(data - reg.predict(x)))
    return loss, coef

def single_attr_spline(data, spline_num):
    ret = np.zeros(spline_num * 2)
    loss_tmp = np.zeros((len(data), len(data)))
    coef_tmp = np.zeros((len(data), len(data)))
    dp_loss = np.zeros((spline_num, len(data), 2))
    dp_loss[:,:,0] = np.inf
    dp_loss[:,:,1] = 0
    
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            curr_loss, curr_coef = get_linreg(data[i:j])
            loss_tmp[i, j] = curr_loss
            coef_tmp[i, j] = curr_coef
    
    for i in range(1, len(data)):
        dp_loss[0, i, 0] = loss_tmp[0, i]
        dp_loss[0, i, 1] = 0
    
    for spline in range(1, spline_num):
        for i in range(spline + 1, len(data)):
            dp_loss[spline, i, 0] = dp_loss[spline - 1, i, 0]
            dp_loss[spline, i, 1] = dp_loss[spline - 1, i, 1]
            for j in range(i):
                if dp_loss[spline - 1, j, 0] + loss_tmp[j, i] < dp_loss[spline, i, 0]:
                    dp_loss[spline, i, 0] = dp_loss[spline - 1, j, 0] + loss_tmp[j, i]
                    dp_loss[spline, i, 1] = j
    
    idx = spline_num - 1
    index = len(data) - 1
    while index > 0:
        ret[idx * 2] = coef_tmp[int(dp_loss[idx, index, 1]), index]
        ret[idx * 2 + 1] = index - dp_loss[idx, index, 1] + 1
        index = int(dp_loss[idx, index, 1])
        idx -= 1
        
    #return list(ret) + [data.iloc[-1]]
    return list(ret) + [np.mean(data), np.std(data), np.median(data), data[-1]]

def single_row(data, attr_list, use_spline = False, spline_num = 2):
    ret = []
    for attr in attr_list:
        res = None
        if use_spline:
            res = single_attr_spline(list(data[attr]), spline_num)
        else:
            res = single_attr_stat(list(data[attr]))
        ret += res
        
    return ret

def data_cleaning(data, lo, hi, use_spline = False, attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"], window_len = 50, spline_num = 2):
    stat_list = ["mean", "stdv", "median", "last", "diff"]
    if use_spline:
        names = ["CryptoType", "Date", "Close**"] + [c + "." + s for c in attr_list for s in ([b + "." + str(a + 1) for a in range(spline_num) for b in ["Slope", "Length"]] + ["mean", "stdv", "median", "last"])]
    else:
        names = ["CryptoType", "Date", "Close**"] + [a + "." + b for a in attr_list for b in stat_list]
    ret = pd.DataFrame(columns=names)
    dct = collections.defaultdict(list)
    for i in tqdm(range(lo, hi)):
        crypto_type = data["CryptoType"].iloc[i]
        crypto_date = data["Date"].iloc[i]
        crypto_close = data["Close**"].iloc[i]
        earliest_date = datetime.strftime(datetime.strptime(crypto_date, "%Y-%m-%d") - timedelta(days=window_len), "%Y-%m-%d")
        subdata = data[(data["CryptoType"] == crypto_type) & (data["Date"] < crypto_date) & (data["Date"] >= earliest_date)]
        if subdata.shape[0] > window_len / 5:
            row = single_row(subdata, attr_list, use_spline, spline_num)
            dct[names[0]] += [crypto_type]
            dct[names[1]] += [crypto_date]
            dct[names[2]] += [crypto_close]
            for j in range(len(names) - 3):
                dct[names[j + 3]] += [row[j]]
                
    return dct, names

def data_cleaning_parallel(data, use_spline = False, attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"], window_len = 50, spline_num = 2, n_cores = 1):
    if n_cores <= 1:
        dct, names = data_cleaning(data, 0, data.shape[0], use_spline, attr_list, window_len, spline_num)
        return pd.DataFrame.from_dict(dct)[names]
    else:
        batch_size = int(math.ceil(data.shape[0] / n_cores))
        dct_lst = Parallel(n_jobs = n_cores)(
            delayed(data_cleaning)(
                data, i * batch_size, min((i + 1) * batch_size, data.shape[0]), use_spline, attr_list, window_len, spline_num
            ) for i in range(n_cores)
        )
        dct = Counter()
        names = None
        for d, ns in dct_lst:
            dct.update(d)
            names = ns
        return pd.DataFrame.from_dict(dct).sort_values("Date")[names]

def reshaping(data, attr_len):
    base = 3
    try:
        n, d = data.shape
    except:
        n = 1
        d = len(data)
    d1 = attr_len
    d2 = (d - base) // attr_len
    ret = np.zeros((n, d1, d2))
    
    if n > 1:
        for i in range(n):
            for j in range(d1):
                ret[i, j, :] = np.array(list(data.iloc[i, (base + j * d2):(base + (j + 1) * d2)]))
    else:
        data = list(data)
        for j in range(d1):
            ret[0, j, :] = np.array(list(data[(base + j * d2):(base + (j + 1) * d2)]))
    return ret

def get_trading_payoff(Y, Y_hat, transaction_length):
    payoff = 0
    idx = 0
    while idx < len(Y_hat) - transaction_length:
        if Y_hat[idx + transaction_length] > Y_hat[idx]:
            payoff += Y[idx + transaction_length] - Y[idx]
        else:
            payoff -= Y[idx + transaction_length] - Y[idx]
        idx += 1
    return payoff

def training_forecasting(data, lstm_params = {"hidden_size": 1, "activations": ["relu"], "output_dim":[50], "epochs": 1000, "verbose":0}, training_until = "2019-01-01", trading_period = 100, transaction_length_arr = [5], attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"], use_spline = False, spline_num = 5, window_len = 100):
    training_set = data[data["Date"] <= training_until]
    latest_date = datetime.strftime(datetime.strptime(training_until, "%Y-%m-%d") + timedelta(days=trading_period), "%Y-%m-%d")
    #test_set = data[(data["Date"] > training_until) & (data["Date"] <= latest_date)]
    test_set = data[(data["Date"] > training_until)]
    
    X_train = reshaping(training_set, len(attr_list))
    X_test = reshaping(test_set, len(attr_list))
    Y_train = np.array(list(training_set["Close**"]))
    Y_test = np.array(list(test_set["Close**"]))
    
    prev_shape = (X_train.shape[1], X_train.shape[2])
    model = Sequential()
    for i in range(lstm_params["hidden_size"]):
        if i == 0:
            model.add(LSTM(lstm_params["output_dim"][i], activation=lstm_params["activations"][i], input_shape=prev_shape, return_sequences=(i < lstm_params["hidden_size"] - 1)))
        else:
            model.add(LSTM(lstm_params["output_dim"][i], activation=lstm_params["activations"][i], return_sequences=(i < lstm_params["hidden_size"] - 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    
    model.fit(X_train, Y_train, epochs=lstm_params["epochs"], verbose=lstm_params["verbose"])
    
    earliest_date = datetime.strftime(datetime.strptime(training_until, "%Y-%m-%d") - timedelta(days=window_len), "%Y-%m-%d")
    subdata_test_close = list(data[(data["CryptoType"] == data["CryptoType"].iloc[0]) & (data["Date"] <= training_until) & (data["Date"] > earliest_date)]["Close**"])
    
    payoff_arr = []
    for transaction_length in transaction_length_arr:
        Y_test_hat = []
        test_set_copy = test_set.copy()
        len_res = 0
        
        for i in tqdm(range(1, trading_period + 1)):
            curr_date = datetime.strftime(datetime.strptime(training_until, "%Y-%m-%d") + timedelta(days=i), "%Y-%m-%d")
            crypto_close = model.predict(reshaping(test_set.iloc[i - 1], len(attr_list)), verbose=lstm_params["verbose"])[0][0]
            test_set.loc[i - 1, "Close**"] = crypto_close
            subdata_test_close.append(crypto_close)
            if i % transaction_length != 0:
                if use_spline:
                    res = single_attr_spline(list(subdata_test_close[-(window_len+1):-1]), spline_num)
                else:
                    res = single_attr_stat(list(subdata_test_close[-(window_len+1):-1]))
                len_res = len(res)
                for j in range(len(res)):
                    test_set.iloc[i - 1, 3 + j] = res[j]
            else:
                #test_set = test_set_copy.copy()
                for k in range(i - transaction_length + 1, i):
                    for j in range(len_res):
                        test_set.iloc[k - 1, 3 + j] = test_set_copy.iloc[k - 1, 3 + j]
                
            Y_test_hat.append(crypto_close)
        Y_test_hat = np.array(Y_test_hat)
        payoff = get_trading_payoff(Y_test, Y_test_hat, transaction_length)
        payoff_arr.append(payoff)
        
    return payoff_arr

def trading_single(crypto, training_until, trading_period, transaction_length_arr, spline_num, window_len, crypto_stat, crypto_spline):
    print("Working on " + crypto + "...")
    print("   Trading on stat of " + crypto + "...")
    profit_stat_arr = training_forecasting(crypto_stat[crypto_stat["CryptoType"] == crypto], training_until=training_until, trading_period=trading_period, transaction_length_arr=transaction_length_arr, lstm_params = {"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": 5000, "verbose":0}, use_spline=False, window_len=window_len)
    print("   Trading on spline of " + crypto + "...")
    profit_spline_arr = training_forecasting(crypto_spline[crypto_spline["CryptoType"] == crypto], training_until=training_until, trading_period=trading_period, transaction_length_arr=transaction_length_arr, lstm_params = {"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": 5000, "verbose":0}, use_spline=True, spline_num=spline_num, window_len=window_len)
    return [crypto] * len(transaction_length_arr), profit_stat_arr, profit_spline_arr, transaction_length_arr

def tradings_comparison(training_until="2019-01-01", trading_period=100, transaction_length_arr=[5], spline_num=5, n_cores=4, window_len=30, attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"], fname_result=None, training_start=None):
    global crypto_info
    if training_start is not None:
        crypto_info = crypto_info[crypto_info["Date"] >= training_start]
    print("Cleaning Dataframe using stats...")
    crypto_stat = data_cleaning_parallel(crypto_info, use_spline=False, spline_num=spline_num, n_cores=n_cores, window_len=window_len, attr_list=attr_list)
    print("Cleaning Dataframe using spline...")
    if "CryptoInfo" in crypto_info_fname:
        fname = "crypto_spline_until=" + training_until + "_spline=" + str(spline_num) + "_window=" + str(window_len) + "_v2.csv"
    else:
        fname = "exg_spline_until=" + training_until + "_spline=" + str(spline_num) + "_window=" + str(window_len) + "_v2.csv"
    if os.path.isfile(fname):
        crypto_spline = pd.read_csv(fname)
        with open(fname, "r") as f:
            for line in f:
                header_names = line.strip("\n").strip(",").split(",")
                break
        crypto_spline = crypto_spline[header_names]
    else:
        crypto_spline = data_cleaning_parallel(crypto_info, use_spline=True, spline_num=spline_num, n_cores=n_cores, attr_list=attr_list)
        crypto_spline.to_csv(fname)
    cryptos = list(set(crypto_stat["CryptoType"]))
    
    dct = {"CryptoType":[], "Profit_stat":[], "Profit_spline":[], "Transaction_length":[]}
    
    results = Parallel(n_jobs=len(cryptos))(delayed(trading_single)(
        cryptos[i], training_until, trading_period, transaction_length_arr, spline_num, window_len, crypto_stat, crypto_spline
    ) for i in range(len(cryptos)))

    for res in results:
        crypto_arr, profit_stat_arr, profit_spline_arr, transaction_length_arr = res
        dct["CryptoType"] += crypto_arr
        dct["Profit_stat"] += profit_stat_arr
        dct["Profit_spline"] += profit_spline_arr
        dct["Transaction_length"] += transaction_length_arr

    if fname_result is None:
        fname_result = "crypto_trading_until=" + training_until + "_spline=" + str(spline_num) + "_window=" + str(window_len) + "_v2.csv"
    else:
        fname_result = fname_result + "_trading_until=" + training_until + "_spline=" + str(spline_num) + "_window=" + str(window_len) + "_v2.csv"
    d = pd.DataFrame.from_dict(dct)
    d.to_csv(fname_result, index=False)
    crypto_info = crypto_info_copy

attr_list = None
if "CryptoInfo" in crypto_info_fname:
    attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"]
else:
    attr_list = ["Close**"]
tradings_comparison(n_cores=36, training_until="2019-01-01", transaction_length_arr=[1, 5, 15, 30, 50], trading_period=150, attr_list=attr_list, fname_result="exg", training_start="2010-01-01")
