import matplotlib
#matplotlib.use('Agg')
import sys, time, math, os.path
import pandas as pd
import numpy as np
from dateutil import parser
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import warnings
import torch

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
all_info = pd.read_csv("data/AllInfo.csv")
all_info = all_info.sort_values(by=["Type", "Date"])

def daily_return(prices):
    return (prices[1:] - prices[:-1]) / prices[:-1]

def return_to_prices(price0, returns):
    ret = np.zeros(len(returns) + 1)
    ret[0] = price0
    for i in range(1, len(ret)):
        ret[i] = (1 + returns[i - 1]) * ret[i - 1]
    return ret[1:]

def return_to_prices_torch(price0, returns):
    ret = torch.zeros(len(returns) + 1)
    ret[0] = price0
    for i in range(1, len(ret)):
        ret[i] = (1 + returns[i - 1]) * ret[i - 1]
    return ret[1:]

def infer_bin_pos(x, bin_ranges):
    idx = 0
    while idx < len(bin_ranges) - 1:
        if x >= bin_ranges[idx + 1]:
            idx += 1
        else:
            break
    return min(idx, len(bin_ranges) - 2)

def infer_n_grams(prices, bin_ranges, N=1):
    n_grams = []
    for i in range(N):
        n_grams.append({})
    for i in tqdm(range(len(prices) - 1)):
        for j in range(1, N + 1):
            if i < len(prices) - j:
                y = infer_bin_pos(prices[i + j], bin_ranges)
                if j > 1:
                    tup = [infer_bin_pos(x, bin_ranges) for x in prices[i:(i + j)]]
                else:
                    tup = [infer_bin_pos(prices[i], bin_ranges)]
                tup = tuple(tup)
                if tup not in n_grams[j - 1]:
                    n_grams[j - 1][tup] = [0] * (len(bin_ranges) - 1)
                n_grams[j - 1][tup][y] += 1
    return n_grams

def get_distribution(curr_idx, prices, n_grams, bin_ranges):
    idx = len(n_grams) - 1
    ret = None
    while idx >= 0:
        tup = prices[(curr_idx - idx - 1):(curr_idx)]
        if idx == 0:
            tup = [tup]
        tup = tuple([infer_bin_pos(x, bin_ranges) for x in tup])
        if tup in n_grams[idx]:
            ret = np.array(n_grams[idx][tup])
            break
        idx -= 1
    if ret is None:
        ret = np.ones(len(bin_ranges) - 1)
    return ret / np.sum(ret)

def n_gram_RL_single(training_set, n=1, num_itr=100, num_repeat=10, step_size=0.01):
    training_return = daily_return(training_set)
    bin_ranges = np.percentile(training_return, np.arange(11) * 10)
    bin_values = (bin_ranges[:-1] + bin_ranges[1:]) / 2
    n_gram_rules = {}
    size = len(bin_values)
    for i in range(n, len(training_return)):
        tup = tuple([infer_bin_pos(x, bin_ranges) for x in training_return[(i - n):i]])
        if tup not in n_gram_rules:
            n_gram_rules[tup] = torch.ones(size, requires_grad=True)
            n_gram_rules[tup].data = n_gram_rules[tup].data / size
    for _ in tqdm(range(num_itr)):
        #loss = torch.tensor(0).double()
        #for _ in range(num_repeat):
        curr_returns = torch.zeros(len(training_return) - n)
        for i in range(n, len(training_return)):
            pos = i - n
            tup = tuple([infer_bin_pos(x, bin_ranges) for x in training_return[(i - n):i]])
            dist = n_gram_rules[tup].data
            val = np.random.choice(bin_values, size=1, p=dist)
            curr_returns[pos] = val[0]
        pred_prices = return_to_prices_torch(training_set[n - 1], curr_returns)
        loss = torch.mean(torch.square(pred_prices - training_set[(n + 1):]))
        loss.backward()
        for tup in n_gram_rules:
            n_gram_rules[tup].data = n_gram_rules[tup].data - step_size * n_gram_rules[tup].grad
            n_gram_rules[tup].grad.detach()
            n_gram_rules[tup].zero_()
    return n_gram_rules

def sim_n_grams(stock_name, n=1, num_itr=1000, training_begin=None, training_until="2019-01-01", test_until="2020-01-01"):
    if training_begin is None:
        curr_info = all_info[all_info["Type"] == stock_name]
    else:
        curr_info = all_info[(all_info["Type"] == stock_name) & (all_info["Date"] >= training_begin)]
    data = np.array(curr_info[curr_info["Date"] <= test_until]["Close**"])
    training_set = np.array(curr_info[curr_info["Date"] <= training_until]["Close**"])
    training_return = daily_return(training_set)
    test_start = len(training_set)
    bin_ranges = np.percentile(training_return, np.arange(11) * 10)
    bin_values = (bin_ranges[:-1] + bin_ranges[1:]) / 2
    
    print("Training...")
    #n_grams = infer_n_grams(training_return, bin_ranges, n)
    n_grams = []
    for i in range(1, n + 1):
        n_grams.append(n_gram_RL_single(training_set, n=i))
    
    print("Simulating on test set...")
    results = np.zeros((num_itr, len(data) - len(training_set)))
    for i in tqdm(range(num_itr)):
        prices = training_return
        for j in range(test_start, len(data)):
            pos = j - test_start
            dist = get_distribution(j - 1, prices, n_grams, bin_ranges)
            val = np.random.choice(bin_values, size=1, p=dist)
            results[i, pos] = val
            prices = np.append(prices, val)
    
    for i in range(num_itr):
        results[i,:] = return_to_prices(training_set[-1], results[i,:])
    
    return results, training_set, data[test_start:], [parser.parse(x) for x in list(curr_info[curr_info["Date"] <= test_until]["Date"])]

def visualize(results, Y_train, Y_test, Date, stock_name, n, alpha=0.05):
    lower_bound = np.percentile(results, int(alpha * 100), axis=0)
    upper_bound = np.percentile(results, int((1 - alpha) * 100), axis=0)
    median_pred = np.percentile(results, 50, axis=0)
    
    train_idx = Date[:len(Y_train)]
    test_idx = Date[len(Y_train):]
    
    plt.plot(train_idx, Y_train, label="Training Values")
    plt.plot(test_idx, Y_test, label="True Test Values")
    plt.plot(test_idx, median_pred, label="Predicted Test Values")
    plt.fill_between(test_idx, lower_bound, upper_bound)
    plt.xlabel("Date")
    plt.ylabel("Prices")
    plt.title(str(n) + "-Gram Model -- " + stock_name)
    plt.legend()
    plt.show()

stock_name = "S&P500"
n = 2
results, training_set, test_set, Date = sim_n_grams(stock_name, n=n, training_begin="2015-01-01")
visualize(results, training_set, test_set, Date, stock_name, n=n, alpha=0.05)
