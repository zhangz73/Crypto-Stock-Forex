import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa import stattools

pd.set_option('display.max_columns', None)

d = pd.read_csv("data/AllInfo_wide_Crypto4.csv")
d2 = d.copy()
names = d.columns.values
N = d.shape[0]
T = (np.arange(N) + 1).reshape(N, 1)
for name in names:
    if name != "Date":
        lm = LinearRegression()
        lm.fit(T, np.array(d[name]))
        res = np.array(d[name]) - lm.predict(T)
        d2[name] = pd.Series(res, index=d2.index)
d2.to_csv("data/AllInfo_wide_Crypto4_residuals.csv", index=False)
