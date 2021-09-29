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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

crypto_info = pd.read_csv("data/CryptoInfo.csv")

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
                
    return dct

def data_cleaning_parallel(data, use_spline = False, attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"], window_len = 50, spline_num = 2, n_cores = 1):
    if n_cores <= 1:
        return pd.DataFrame.from_dict(data_cleaning(data, 0, data.shape[0], use_spline, attr_list, window_len, spline_num))
    else:
        batch_size = int(math.ceil(data.shape[0] / n_cores))
        dct_lst = Parallel(n_jobs = n_cores)(
            delayed(data_cleaning)(
                data, i * batch_size, min((i + 1) * batch_size, data.shape[0]), use_spline, attr_list, window_len, spline_num
            ) for i in range(n_cores)
        )
        dct = Counter()
        for d in dct_lst:
            dct.update(d)
        return pd.DataFrame.from_dict(dct).sort_values("Date")

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
                ret[i, j, :] = list(data.iloc[i, (base + j * d2):(base + (j + 1) * d2)])
    else:
        data = list(data)
        for j in range(d1):
            ret[0, j, :] = list(data[(base + j * d2):(base + (j + 1) * d2)])
    return ret

def training_forecasting(data, lstm_params = {"hidden_size": 1, "activations": ["relu"], "output_dim":[50], "epochs": 1000, "verbose":0}, training_until = "2019-01-01", test_size = 50, attr_list = ["Close**", "Volume", "Market Cap", "MarketShare"], use_spline = False, spline_num = 5, window_len = 100):
    training_set = data[data["Date"] <= training_until]
    latest_date = datetime.strftime(datetime.strptime(training_until, "%Y-%m-%d") + timedelta(days=test_size), "%Y-%m-%d")
    test_set = data[(data["Date"] > training_until) & (data["Date"] <= latest_date)]
    
    X_train = reshaping(training_set, len(attr_list))
    X_test = reshaping(test_set, len(attr_list))
    Y_train = list(training_set["Close**"])
    Y_test = list(test_set["Close**"])
    
    Date = list(training_set["Date"])
    IsTrain = [1] * training_set.shape[0]
    
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
    
    Y_train_hat = model.predict(X_train, verbose=lstm_params["verbose"])
    Y_train_hat = Y_train_hat.reshape(Y_train_hat.shape[0])
    
    earliest_date = datetime.strftime(datetime.strptime(training_until, "%Y-%m-%d") - timedelta(days=window_len), "%Y-%m-%d")
    subdata_test_close = list(data[(data["CryptoType"] == data["CryptoType"].iloc[0]) & (data["Date"] <= training_until) & (data["Date"] > earliest_date)]["Close**"])
    Y_test_hat = []
    for i in range(1, test_size + 1):
        curr_date = datetime.strftime(datetime.strptime(training_until, "%Y-%m-%d") + timedelta(days=i), "%Y-%m-%d")
        crypto_close = model.predict(reshaping(test_set.iloc[i - 1], len(attr_list)), verbose=lstm_params["verbose"])[0][0]
        test_set.loc[i - 1, "Close**"] = crypto_close
        subdata_test_close.append(crypto_close)
        Date.append(curr_date)
        IsTrain.append(0)
        if use_spline:
            res = single_attr_spline(list(subdata_test_close[-(window_len+1):-1]), spline_num)
        else:
            res = single_attr_stat(list(subdata_test_close[-(window_len+1):-1]))
        for j in range(len(res)):
            test_set.iloc[i - 1, 3 + j] = res[j]
            
        Y_test_hat.append(crypto_close)
    Y_test_hat = np.array(Y_test_hat)
        
    #Y_test_hat = model.predict(X_test, verbose=lstm_params["verbose"])
    
#    plt.plot(list(range(len(Y_train))), Y_train, label="True Values for Training Set")
#    plt.plot(list(range(len(Y_train_hat))), Y_train_hat, label="Fitted Values for Training Set")
#    plt.plot(list(range(len(Y_train) - 1, len(Y_train) + len(Y_test))), [Y_train[-1]] + list(Y_test), label="True Values for Test Set")
#    plt.plot(list(range(len(Y_train_hat) - 1, len(Y_train_hat) + len(Y_test_hat))), [Y_train_hat[-1]] + list(Y_test_hat), label="Fitted Values for Test Set")
#    plt.legend()
#    plt.show()
    
#    return np.mean(np.square(Y_train - Y_train_hat)), np.mean(np.square(Y_test - Y_test_hat))
    return list(Y_train) + list(Y_test), list(Y_train_hat) + list(Y_test_hat), Date, IsTrain, [data["CryptoType"].iloc[0]] * len(IsTrain)

def models_comparison(training_until="2019-01-01", test_size=100, epoch_lst=np.arange(11)*500, spline_num=5, n_cores=4, window_len=30):
    print("Cleaning Dataframe using stats...")
    crypto_stat = data_cleaning_parallel(crypto_info, use_spline=False, spline_num=spline_num, n_cores=n_cores, window_len=window_len)
    print("Cleaning Dataframe using spline...")
    fname = "crypto_spline_until=" + training_until + "_spline=" + str(spline_num) + "_window=" + str(window_len) + "_v2.csv"
    if os.path.isfile(fname):
        crypto_spline = pd.read_csv(fname)
    else:
        crypto_spline = data_cleaning_parallel(crypto_info, use_spline=True, spline_num=spline_num, n_cores=n_cores)
        crypto_spline.to_csv(fname)
    cryptos = list(set(crypto_stat["CryptoType"]))
    
    dct = {"Y":[], "Y_hat":[], "Date":[], "IsTrain":[], "CryptoType":[], "Epochs":[], "Methodology":[]}
    
    for crypto in cryptos:
        print("Training on " + crypto + "...")
#        train_mse_stat_lst = []
#        test_mse_stat_lst = []
#        train_mse_spline_lst = []
#        test_mse_spline_lst = []
        for epoch in tqdm(epoch_lst):
#            train_mse_stat, test_mse_stat = training_forecasting(crypto_stat[crypto_stat["CryptoType"] == crypto], training_until=training_until, test_size=test_size, lstm_params = {"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": epoch, "verbose":0}, use_spline=False, window_len=window_len)
#            train_mse_spline, test_mse_spline = training_forecasting(crypto_spline[crypto_spline["CryptoType"] == crypto], training_until=training_until, test_size=test_size, lstm_params = {"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": epoch, "verbose":0}, use_spline=True, spline_num=spline_num, window_len=window_len)
            Y_stat, Y_hat_stat, Date_stat, IsTrain_stat, Type_stat = training_forecasting(crypto_stat[crypto_stat["CryptoType"] == crypto], training_until=training_until, test_size=test_size, lstm_params = {"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": epoch, "verbose":0}, use_spline=False, window_len=window_len)
            Y_spline, Y_hat_spline, Date_spline, IsTrain_spline, Type_spline = training_forecasting(crypto_spline[crypto_spline["CryptoType"] == crypto], training_until=training_until, test_size=test_size, lstm_params = {"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": epoch, "verbose":0}, use_spline=True, spline_num=spline_num, window_len=window_len)
            dct["Y"] += Y_stat + Y_spline
            dct["Y_hat"] += Y_hat_stat + Y_hat_spline
            dct["Date"] += Date_stat + Date_spline
            dct["IsTrain"] += IsTrain_stat + IsTrain_spline
            dct["CryptoType"] += Type_stat + Type_spline
            dct["Epochs"] += [epoch] * (len(Y_stat) + len(Y_spline))
            dct["Methodology"] += ["stat"] * len(Y_stat) + ["spline"] * len(Y_spline)
            
#            train_mse_stat_lst.append(train_mse_stat)
#            test_mse_stat_lst.append(test_mse_stat)
#            train_mse_spline_lst.append(train_mse_spline)
#            test_mse_spline_lst.append(test_mse_spline)
#
#        plt.plot(epoch_lst, train_mse_stat_lst, label="Training MSE Using Stats")
#        plt.plot(epoch_lst, test_mse_stat_lst, label="Test MSE Using Stats")
#        plt.plot(epoch_lst, train_mse_spline_lst, label="Training MSE Using Spline")
#        plt.plot(epoch_lst, test_mse_spline_lst, label="Test MSE Using Spline")
#        plt.title("Stats VS Spline -- " + crypto)
#        plt.xlabel("Training Epochs")
#        plt.ylabel("MSE")
#        plt.legend()
#        plt.savefig("comparison_plots/" + crypto + ".png")
#        plt.clf()
    fname_result = "crypto_results_until=" + training_until + "_spline=" + str(spline_num) + "_window=" + str(window_len) + "_v2.csv"
    d = pd.DataFrame.from_dict(dct)
    d.to_csv(fname_result, index=False)

models_comparison(n_cores=36, training_until="2019-01-01")
models_comparison(n_cores=36, training_until="2020-01-01")
