from bs4 import BeautifulSoup
import requests, sys, time
import csv
import pandas as pd
from datetime import datetime
import random
from itertools import cycle
from lxml.html import fromstring
import warnings

warnings.filterwarnings("ignore")

user_agent_list = [
   #Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
]

crypto_list = ["Bitcoin", "Ethereum", "XRP", "Bitcoin-Cash", "Bitcoin-SV", "Litecoin", "Binance-Coin", "EOS", "Cardano", "Tezos", "Stellar", "Monero", "TRON", "Neo", "Ethereum-Classic"]

def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:50]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            #Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies

def get_crypto(crypto_name, content=None):
    print("Collecting " + crypto_name + "...")
    if content is None:
        url = "https://coinmarketcap.com/currencies/" + crypto_name + "/historical-data/?start=20120101&end=" + datetime.now().strftime("%Y-%m-%d").replace("-", "")
        output_name = "data/" + crypto_name + ".csv"
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
    else:
        soup = BeautifulSoup(content)
    rows = soup.find_all('tr')
    content = []

    nrow = 0
    for row in rows:
        nrow += 1
        if nrow > 2:
            curr_row = []
            idx = 0
            if len(row.find_all('td')) > 2:
                for td in row.find_all('td'):
                    idx += 1
                    data = td.find('div')
                    data = td.getText()
                    if idx == 1:
                        data = datetime.strptime(data, "%b %d, %Y").strftime("%Y-%m-%d")
                    data = data.replace(",", "")
                    curr_row.append(str(data))
            if len(row.find_all('th')) > 2:
                for th in row.find_all('th'):
                    data = th.getText()
                    curr_row.append(str(data))
            if len(curr_row) > 0:
                content.append(curr_row)
    
    ret = [["CryptoType"] + content[0]] + [[crypto_name] + x for x in reversed(content[1:])]
    return ret

def get_market_cap():
    print("Collecting Total Market Capitalizations...")
    all_dates = [x.strftime("%Y-%m-%d") for x in pd.date_range(start="2013-04-28", end=datetime.now().strftime("%Y-%m-%d"))]
    output_name = "data/marketcap.csv"
    content = [["Date", "TotalMarketCap"]]
    idx = 0
    proxies = get_proxies()
    print(proxies)
    proxy_pool = cycle(proxies)
    
    proxy_idx = 0
    
    for d in all_dates:
        print("We are at " + d, end="\r")
        url = "https://coinmarketcap.com/historical/" + d.replace("-", "")
        user_agent = random.choice(user_agent_list)
        headers = {'User-Agent': 'Chrome'}
        proxy = next(proxy_pool)
        proxy_idx += 1
        #Make the request
        passing = False
        while not passing:
            try:
                #response = requests.get(url, proxies={"http": "http://"+proxy, "https": "https://" + proxy}, verify=False, headers=headers)
                response = requests.get(url)
                soup = BeautifulSoup(response.content)
                row = soup.find_all('strong')
                if len(row) > 0:
                    row = row[0]
                    curr_cap = row.getText().split("$")[1].replace(",", "")
                    content.append([str(d), str(curr_cap)])
                    time.sleep(3)
                passing = True
                if len(row) == 0:
                    passing = False
            except:
                print("Some errors occurred. Retrying...")
            if not passing:
                nap = 15
                print("Collecting too fast. Sleeping for " + str(nap) + " secs...")
                time.sleep(nap)
                """
                if proxy_idx >= len(proxies):
                    proxy_idx = 0
                    proxies = get_proxies()
                    proxy_pool = cycle(proxies)
                    print("Proxy pool refreshed at " + str(d))
                else:
                    proxy = next(proxy_pool)
                    proxy_idx += 1
                    print("Proxy " + str(proxy) + " doesn't work. Trying the next one...")
                """
        idx += 1
    print("")
    with open(output_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(content)

def combine_results():
    print("Building the combined dataset...")
    cryptoPrices = pd.read_csv("data/CryptoPrices.csv")
    marketCap = pd.read_csv("data/marketcap.csv")
    comb = cryptoPrices.join(marketCap.set_index("Date"), on="Date", how="inner")
    comb["MarketShare"] = comb["Market Cap"] / comb["TotalMarketCap"] * 100
    comb.to_csv("data/CryptoInfo.csv", index=False)

"""
content_all = []
for crypto_name in crypto_list:
    passing = False
    while not passing:
        try:
            res = get_crypto(crypto_name)
            passing = True
        except:
            nap = 30
            print("Collecting too fast. Sleeping for " + str(nap) + " secs...")
            time.sleep(nap)
    if len(content_all) == 0:
        content_all.append(res[0])
    content_all += res[1:]
    
with open("data/CryptoPrices.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(content_all)
"""
#get_market_cap()
#combine_results()
#print("Jobs Done!")
