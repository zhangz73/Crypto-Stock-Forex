import matplotlib
matplotlib.use('Agg')
import sys, time, math, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import dateutil.parser as parser

pd.set_option('display.max_columns', None)

result_name = sys.argv[1]
dir = result_name.replace(".csv", "")

crypto_info = pd.read_csv(result_name)

if not os.path.isdir(dir):
    os.mkdir(dir)

def parseDate(dates):
    #return [datetime.strptime(date, "%m/%d/%y") for date in dates]
    return [parser.parse(date) for date in dates]

def plot_single_forecast(CryptoType, Epoch):
    subdata_stat = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Epochs"] == Epoch) & (crypto_info["Methodology"] == "stat")]
    subdata_spline = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Epochs"] == Epoch) & (crypto_info["Methodology"] == "spline")]
    
    train_test_boundary = np.argmin(subdata_stat["IsTrain"])
    
    plt.plot(parseDate(subdata_stat["Date"]), subdata_stat["Y"], label="True Prices", color="red")
    plt.plot(parseDate(subdata_stat["Date"]), subdata_stat["Y_hat"], label="Predicted Prices Using Trend")
    plt.plot(parseDate(subdata_spline["Date"]), subdata_spline["Y_hat"], label="Predicted Prices Using Spline")
    #plt.axvline(datetime.strptime(subdata_stat["Date"].iloc[train_test_boundary], "%m/%d/%y"), label="Train/Test Boundary", color="black")
    plt.axvline(parser.parse(subdata_stat["Date"].iloc[train_test_boundary]), label="Train/Test Boundary", color="black")
    plt.title("Forecasting on " + CryptoType + " with " + str(Epoch) + " epochs")
    plt.legend()
    plt.savefig(dir + "/" + CryptoType + "_" + str(Epoch) + "_forecast.png")
    plt.clf()

def plot_single_residual(CryptoType, Epoch):
    subdata_stat = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Epochs"] == Epoch) & (crypto_info["Methodology"] == "stat")]
    subdata_spline = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Epochs"] == Epoch) & (crypto_info["Methodology"] == "spline")]
    
    train_test_boundary = np.argmin(subdata_stat["IsTrain"])
    
    MSE_train_stat = np.mean(np.square(subdata_stat[subdata_stat["IsTrain"] == 1]["Y_hat"] - subdata_stat[subdata_stat["IsTrain"] == 1]["Y"]))
    MSE_test_stat = np.mean(np.square(subdata_stat[subdata_stat["IsTrain"] == 0]["Y_hat"] - subdata_stat[subdata_stat["IsTrain"] == 0]["Y"]))
    MSE_train_spline = np.mean(np.square(subdata_spline[subdata_spline["IsTrain"] == 1]["Y_hat"] - subdata_spline[subdata_spline["IsTrain"] == 1]["Y"]))
    MSE_test_spline = np.mean(np.square(subdata_spline[subdata_spline["IsTrain"] == 0]["Y_hat"] - subdata_spline[subdata_spline["IsTrain"] == 0]["Y"]))
    
    plt.plot(parseDate(subdata_stat["Date"]), subdata_stat["Y_hat"] - subdata_stat["Y"], label="Residuals Using Trend")
    plt.plot(parseDate(subdata_spline["Date"]), subdata_spline["Y_hat"] - subdata_spline["Y"], label="Residuals Using Spline")
    #plt.axvline(datetime.strptime(subdata_stat["Date"].iloc[train_test_boundary], "%m/%d/%y"), label="Train/Test Boundary", color="black")
    plt.axvline(parser.parse(subdata_stat["Date"].iloc[train_test_boundary]), label="Train/Test Boundary", color="black")
    plt.title("Forecasting on " + CryptoType + " with " + str(Epoch) + " epochs\n" + "Trend: MSE_train = " + str(MSE_train_stat) + "; MSE_test = " + str(MSE_test_stat) + "\n" + "Spline: MSE_train = " + str(MSE_train_spline) + "; MSE_test = " + str(MSE_test_spline))
    plt.legend()
    plt.savefig(dir + "/" + CryptoType + "_" + str(Epoch) + "_residual.png")
    plt.clf()

def plot_single_cumMSE(CryptoType, Epoch):
    subdata_stat = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Epochs"] == Epoch) & (crypto_info["Methodology"] == "stat")]
    subdata_spline = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Epochs"] == Epoch) & (crypto_info["Methodology"] == "spline")]
    
    train_test_boundary = np.argmin(subdata_stat["IsTrain"])
    
    MSE_test_stat = np.cumsum(np.square(subdata_stat[subdata_stat["IsTrain"] == 0]["Y_hat"] - subdata_stat[subdata_stat["IsTrain"] == 0]["Y"]))
    MSE_test_stat = MSE_test_stat / (np.arange(len(MSE_test_stat)) + 1)
    MSE_test_spline = np.cumsum(np.square(subdata_spline[subdata_spline["IsTrain"] == 0]["Y_hat"] - subdata_spline[subdata_spline["IsTrain"] == 0]["Y"]))
    MSE_test_spline = MSE_test_spline / (np.arange(len(MSE_test_spline)) + 1)
    
    plt.plot(parseDate(subdata_stat[subdata_stat["IsTrain"] == 0]["Date"]), MSE_test_stat, label="Residuals Using Trend")
    plt.plot(parseDate(subdata_spline[subdata_spline["IsTrain"] == 0]["Date"]), MSE_test_spline, label="Residuals Using Spline")
    plt.title("Cumulative Test MSE on " + CryptoType + " with " + str(Epoch) + " epochs")
    plt.legend()
    plt.savefig(dir + "/" + CryptoType + "_" + str(Epoch) + "_cumMSE.png")
    plt.clf()

def plot_single_epoch(CryptoType):
    epochs = list(set(crypto_info["Epochs"]))
    epochs.sort()
    
    subdata_stat = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Methodology"] == "stat")]
    subdata_spline = crypto_info[(crypto_info["CryptoType"] == CryptoType) & (crypto_info["Methodology"] == "spline")]
    
    MSE_train_stat_arr = []
    MSE_test_stat_arr = []
    MSE_train_spline_arr = []
    MSE_test_spline_arr = []
    
    for epoch in epochs:
        MSE_train_stat = np.mean(np.square(subdata_stat[(subdata_stat["IsTrain"] == 1) & (subdata_stat["Epochs"] == epoch)]["Y_hat"] - subdata_stat[(subdata_stat["IsTrain"] == 1) & (subdata_stat["Epochs"] == epoch)]["Y"]))
        MSE_test_stat = np.mean(np.square(subdata_stat[(subdata_stat["IsTrain"] == 0) & (subdata_stat["Epochs"] == epoch)]["Y_hat"] - subdata_stat[(subdata_stat["IsTrain"] == 0) & (subdata_stat["Epochs"] == epoch)]["Y"]))
        MSE_train_spline = np.mean(np.square(subdata_spline[(subdata_spline["IsTrain"] == 1) & (subdata_spline["Epochs"] == epoch)]["Y_hat"] - subdata_spline[(subdata_spline["IsTrain"] == 1) & (subdata_spline["Epochs"] == epoch)]["Y"]))
        MSE_test_spline = np.mean(np.square(subdata_spline[(subdata_spline["IsTrain"] == 0) & (subdata_spline["Epochs"] == epoch)]["Y_hat"] - subdata_spline[(subdata_spline["IsTrain"] == 0) & (subdata_spline["Epochs"] == epoch)]["Y"]))
        
        MSE_train_stat_arr.append(MSE_train_stat)
        MSE_test_stat_arr.append(MSE_test_stat)
        MSE_train_spline_arr.append(MSE_train_spline)
        MSE_test_spline_arr.append(MSE_test_spline)
    
    plt.plot(epochs, MSE_train_stat_arr, label="Training Set Using Trend")
    plt.plot(epochs, MSE_test_stat_arr, label="Test Set Using Trend")
    plt.plot(epochs, MSE_train_spline_arr, label="Training Set Using Spline")
    plt.plot(epochs, MSE_test_spline_arr, label="Test Set Using Spline")
    plt.title("MSE of " + CryptoType + " across different epochs")
    plt.legend()
    plt.savefig(dir + "/" + CryptoType + "_epochs.png")
    plt.clf()

crypto_types = list(set(crypto_info["CryptoType"]))
epochs = list(set(crypto_info["Epochs"]))

for CryptoType in crypto_types:
    for Epoch in epochs:
        plot_single_forecast(CryptoType, Epoch)
        plot_single_residual(CryptoType, Epoch)
        plot_single_cumMSE(CryptoType, Epoch)
    plot_single_epoch(CryptoType)
