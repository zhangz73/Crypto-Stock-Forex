import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa import stattools

pd.set_option('display.max_columns', None)

d = pd.read_csv("data/MixInfo.csv")[["Type", "Date", "Close**"]]
d2 = pd.read_csv("data/CryptoInfo.csv")[["CryptoType", "Date", "Close**"]]
d2.columns = ["Type", "Date", "Close**"]
d3 = pd.read_csv("data/ExgInfo.csv")[["CryptoType", "Date", "Close**"]]
d3 = d3[d3["CryptoType"].isin(["aud2usd", "cad2usd", "jpy2usd"])]
d3.columns = ["Type", "Date", "Close**"]
d = d.append(d2)
d = d.append(d3)
d = d.sort_values(by=["Type", "Date"])
#d.to_csv("data/AllInfo.csv", index=False)
d = d[d["Type"].isin(["cad2usd", "EUR2USD", "Ethereum", "Litecoin", "Stellar", "aud2usd", "CNY2USD", "jpy2usd", "S&P500", "Bitcoin", "GBP2USD", "VIX", "GOLD"]
)]
#d = d[d["Date"] >= "2017-07-01"]
d2 = d2[d2["Type"].isin(["Bitcoin", "Ethereum", "Stellar", "Bitcoin-Cash", "XRP", "Litecoin", "TRON"])]

def long_to_wide(d):
    types = list(set(d["Type"]))
    ret_lst = [d[d["Type"] == tp] for tp in types]
    ret = ret_lst[0][["Date", "Close**"]]
    type0 = str(list(set(ret_lst[0]["Type"]))[0])
    ret.columns = ["Date", "Close**" + type0]
    for i in range(1, len(ret_lst)):
        typei = str(list(set(ret_lst[i]["Type"]))[0])
        tmp = ret_lst[i][["Date", "Close**"]]
        tmp.columns = ["Date", "Close**" + typei]
        ret = ret.merge(tmp, on="Date")
    ret.columns = [x.replace("**", "_") for x in ret.columns]
    return ret

def plot_all(d):
    dates = [parser.parse(x) for x in d["Date"]]
    cols = [x for x in d.columns if x != "Date"]
    plt.figure(figsize=(10, 10))
    for col in cols:
        plt.plot(dates, list(d[col]), label=col.split("_")[-1])
    plt.xlabel("Time")
    plt.ylabel("Prices")
    plt.yscale("log")
    plt.legend()
    plt.title("Prices over time")
    plt.savefig("MixPlotsAll_Crypto.png")
    plt.clf()
    
    d_copy = d.copy()
    d_copy.columns = [x.replace("Close_", "") for x in d.columns]
    corr = d_copy.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, annot=True, fmt='.2f',
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    ax.set_xticklabels(ax.get_xticklabels(), rotation="vertical")
    plt.title("Correlation Matrix")
    plt.savefig("MixCorrPlot_Crypto.png")
    plt.clf()

def get_residuals(ret, attr_lst, fname):
    title = attr_lst[-1] + " ~ " + " + ".join(attr_lst[:-1])
    attr_lst = ["Close_" + x for x in attr_lst]
    data = ret[attr_lst]
    X = np.array(data[attr_lst[:-1]])
    Y = np.array(data[attr_lst[-1]])
    reg = LinearRegression().fit(X, Y)
    loss = Y - reg.predict(X)
    dates = [parser.parse(x) for x in ret["Date"]]
    MSE = np.mean(np.square(loss))
    p_val = stattools.kpss(loss)[1]
    
    plt.plot(dates, loss)
    plt.axhline(color="red")
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.title(title + "\nMSE = " + str(MSE) + "; KPSS p-value = " + str(p_val))
    plt.savefig(fname)
    plt.clf()
    
    return MSE

ret = long_to_wide(d2)
ret.to_csv("data/CryptoInfo_wide_Crypto7.csv", index=False)
#plot_all(ret)
#print(get_residuals(ret, ["GOLD", "VIX", "S&P500", "GBP2USD", "CNY2USD"], "ResPlot_noEUR.png"))
#print(get_residuals(ret, ["EUR2USD", "CNY2USD"], "ResPlot_EUR.png"))
#print(get_residuals(ret, ["GOLD", "CNY2USD", "Bitcoin"], "ResPlot_Bitcoin.png"))
