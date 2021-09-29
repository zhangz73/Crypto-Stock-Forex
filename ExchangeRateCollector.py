from bs4 import BeautifulSoup
import requests, sys, time
import csv
import pandas as pd
from datetime import datetime
from os import listdir
from os.path import isfile, join
import warnings

warnings.filterwarnings("ignore")

def get_exg(fpath):
    title_arr = [f.replace(".txt", "") for f in listdir(fpath) if isfile(join(fpath, f)) and f[0] != "."]
    output_name = "data/ExgInfo.csv"
    dct = {"CryptoType":[], "Date":[], "Close**":[]}
    for title in title_arr:
        print("Collecting " + title + "...")
        url = fpath
        with open(fpath + "/" + title + ".txt", "r") as f:
            content = f.read()
        soup = BeautifulSoup(content)
        table = soup.find_all('tbody')[-1]
        rows = table.find_all("tr")
        for row in rows:
            td_arr = row.find_all('td')
            Date = td_arr[0].getText()
            rate = td_arr[1].getText()
            Date = datetime.strptime(Date, "%B %d, %Y").strftime("%Y-%m-%d")
            dct["Date"].append(Date)
            dct["Close**"].append(rate)
            dct["CryptoType"].append(title)
    d = pd.DataFrame.from_dict(dct)
    d.to_csv(output_name, index=False)

get_exg("exg")
print("Jobs Done!")
