import matplotlib
matplotlib.use('Agg')
import sys, time, math, os.path
import pandas as pd
import numpy as np
from dateutil import parser
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
crypto_info = pd.read_csv("data/AllInfo.csv")

def long_to_wide(d, types):
    print("Reformatting Data...")
    ret_lst = [d[d["Type"] == tp] for tp in types]
    ret = ret_lst[0][["Date", "Close**"]]
    type0 = str(list(set(ret_lst[0]["Type"]))[0])
    ret.columns = ["Date", "Close**" + type0]
    for i in range(1, len(ret_lst)):
        typei = str(list(set(ret_lst[i]["Type"]))[0])
        tmp = ret_lst[i][["Date", "Close**"]]
        tmp.columns = ["Date", "Close**" + typei]
        ret = ret.merge(tmp, on="Date")
    ret.columns = [x.replace("Close**", "").upper() for x in ret.columns]
    for type in types:
        ret[type.upper()] /= np.mean(list(ret[type.upper()]))
    return ret

def reshaping(data):
    try:
        n, d = data.shape
    except:
        n = 1
        d = len(data)
    ret = np.zeros((n, 1, d))

    if n > 1:
        for i in range(n):
            ret[i, 0, :] = data[i,:]
    else:
        ret[0, 0, :] = data
    return ret

def construct_data(data, names, start_index, end_index):
    outputs = []
    for i in range(len(names)):
        ret = np.zeros((end_index - start_index, len(names[i])))
        for j in range(len(names[i])):
            name = names[i][j]
            type = name.split("_")[0].upper()
            offset = int(name.split("_")[1].split("-")[1])
            prices = list(data.iloc[(start_index - offset):(end_index - offset)][type])
            ret[:,j] = np.array(prices)
        outputs.append(ret)
    return outputs

def causal_model_fit(data, inputs_names, outputs_names, lstm_params_lst = [{"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": 5000, "verbose":0}], training_begin="2018-01-01", training_until="2019-01-01"):
    if len(lstm_params_lst) == 1:
        lstm_params_lst = lstm_params_lst * len(inputs_names)
    assert len(lstm_params_lst) == len(inputs_names)
    assert len(inputs_names) == len(outputs_names)
    
    print("Begin Training...")
    idx = 0
    while idx < data.shape[0] and data.iloc[idx]["DATE"] < training_begin:
        idx += 1
    start_index = idx
    while idx < data.shape[0] and data.iloc[idx]["DATE"] < training_until:
        idx += 1
    end_index = idx
    inputs = construct_data(data, inputs_names, start_index, end_index)
    outputs = construct_data(data, outputs_names, start_index, end_index)
    
    predictions = []
    model_pipeline = []
    for i in tqdm(range(len(inputs))):
        model = Sequential()
        if i == 0:
            X = inputs[i]
        elif len(inputs[i]) == 0 or inputs[i].shape[1] == 0:
            X = predictions[i - 1]
        else:
            X = np.hstack((inputs[i], predictions[i - 1]))
        X = reshaping(X)
        prev_shape = (X.shape[1], X.shape[2])
        for j in range(lstm_params_lst[i]["hidden_size"]):
            if j == 0:
                model.add(LSTM(lstm_params_lst[i]["output_dim"][j], activation=lstm_params_lst[i]["activations"][j], input_shape=prev_shape, return_sequences=(j < lstm_params_lst[i]["hidden_size"] - 1)))
            else:
                model.add(LSTM(lstm_params_lst[i]["output_dim"][j], activation=lstm_params_lst[i]["activations"][j], return_sequences=(j < lstm_params_lst[i]["hidden_size"] - 1)))
        model.add(Dense(outputs[i].shape[1]))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, outputs[i], epochs=lstm_params_lst[i]["epochs"], verbose=lstm_params_lst[i]["verbose"])
        Y_hat = model.predict(X, verbose=lstm_params_lst[i]["verbose"])
        model_pipeline.append(model)
        predictions.append(Y_hat)

    return model_pipeline, predictions, outputs

def causal_model_predict(crypto_wide, inputs_names, outputs_names, model_pipeline, start_date="2019-01-01", end_date="2020-01-01"):
    assert len(inputs_names) == len(outputs_names)
    assert len(inputs_names) == len(model_pipeline)
    
    print("Begin Testing...")
    data = crypto_wide.copy()
    #data.columns = [x.replace("Close_", "").upper() for x in data.columns]
    predictions = []
    for i in range(len(outputs_names)):
        predictions.append([])
    idx = 0
    while idx < data.shape[0] and data.iloc[idx]["DATE"] < start_date:
        idx += 1
    start_index = idx
    while idx < data.shape[0] and data.iloc[idx]["DATE"] <= end_date:
        print("   We are at " + str(data.iloc[idx]["DATE"]) + "/" + end_date, end="\r")
        for j in range(len(model_pipeline)):
            curr_input_names = inputs_names[j].copy()
            if j > 0:
                curr_input_names += outputs_names[j - 1]
            X = []
            for name in curr_input_names:
                type = name.split("_")[0].upper()
                offset = int(name.split("_")[1].split("-")[1])
                price = data.iloc[idx - offset][type]
                X.append(price)
            X = reshaping(np.array(X))
            Y_hat = list(model_pipeline[j].predict(X)[0])
            predictions[j].append(Y_hat)
            for i in range(len(outputs_names[j])):
                name = outputs_names[j][i]
                type = name.split("_")[0].upper()
                offset = int(name.split("_")[1].split("-")[1])
                price = Y_hat[i]
                data.iloc[idx - offset][type] = price
        idx += 1
    end_index = idx
    outputs = construct_data(crypto_wide, outputs_names, start_index, end_index)
    for i in range(len(predictions)):
        predictions[i] = np.array(predictions[i])
    return predictions, outputs

def visualize_results(Y, Y_hat, outputs_names, train=False):
    print("Visualizing Result...")
    for i in range(len(outputs_names)):
        for j in range(len(outputs_names[i])):
            name = outputs_names[i][j].split("_")[0].upper()
            if train:
                title = "Model Performance on Training Set -- " + name
                fname = name + "_train.png"
            else:
                title = "Model Performance on Test Set -- " + name
                fname = name + "_test.png"
            X = np.arange(Y[i].shape[0])
            mse = np.mean(np.square(Y[i][:,j] - Y_hat[i][:,j]))
            title += "\nMSE = " + str(mse)
            plt.plot(X, Y[i][:,j], label="True Values")
            plt.plot(X, Y_hat[i][:,j], label="Predicted Values")
            plt.xlabel("Index")
            plt.ylabel("Price")
            plt.title(title)
            plt.legend()
            plt.savefig("CausalModels/" + fname)
            plt.clf()

crypto_wide = long_to_wide(crypto_info, ["Bitcoin", "Bitcoin-Cash", "Ethereum", "Litecoin", "XRP", "Stellar", "TRON"])
inputs_names = [["Bitcoin_t-0", "Bitcoin_t-1", "Bitcoin-Cash_t-0", "Bitcoin-Cash_t-1", "Ethereum_t-0", "Ethereum_t-1", "Litecoin_t-0", "Litecoin_t-1"], ["XRP_t-1", "Stellar_t-0", "TRON_t-1"]]
outputs_names = [["XRP_t-0"], ["TRON_t-0"]]

model_pipeline, Y_hat_train, Y_train = causal_model_fit(crypto_wide, inputs_names, outputs_names, lstm_params_lst = [{"hidden_size": 2, "activations": ["sigmoid", "tanh"], "output_dim":[50, 50], "epochs": 500, "verbose":0}], training_until="2019-01-01")
Y_hat_test, Y_test = causal_model_predict(crypto_wide, inputs_names, outputs_names, model_pipeline, start_date="2019-01-01", end_date="2020-01-01")

visualize_results(Y_train, Y_hat_train, outputs_names, train=True)
visualize_results(Y_test, Y_hat_test, outputs_names, train=False)

print("Jobs Done!")
