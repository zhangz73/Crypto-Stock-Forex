from bs4 import BeautifulSoup
import requests, sys, time
import csv
import pandas as pd
from datetime import datetime
import random
from itertools import cycle
from lxml.html import fromstring
import warnings

def get_gold(fname):
    print("Collecting Gold...")
    content = open(fname, "r").read()
    soup = BeautifulSoup(content)
    rows = soup.find_all('tr')
    content = []
    header = []

    nrow = 0
    for row in rows:
        nrow += 1
        if nrow > 0:
            curr_row = []
            idx = 0
            if len(row.find_all('td')) > 2:
                for td in row.find_all('td'):
                    idx += 1
                    data = td.getText().strip()
                    if idx == 1:
                        try:
                            data = datetime.strptime(data, "%b %d, %Y").strftime("%Y-%m-%d")
                        except:
                            break
                    data = data.replace(",", "").strip("$")
                    curr_row.append(str(data))
            if len(row.find_all('th')) > 2 and len(content) == 0:
                for th in row.find_all('th'):
                    data = th.getText()
                    header.append(str(data))
                print(header)
            if len(curr_row) > 0:
                content.append(curr_row)
    
    return [header] + content[::-1]

ret = get_gold("Gold_1d.txt")
with open("Gold.csv", "w") as f:
    for row in ret:
        f.write(",".join(row) + "\n")
