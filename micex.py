import pandas as pd
import requests

from bs4 import BeautifulSoup

#sample urls
url = ["https://iss.moex.com/iss/history/engines/stock/markets/index/securities/RUPCI.xml?iss.meta=on&from=2019-04-22&till=2020-05-22&limit=100&start=100&sort_order=TRADEDATE&sort_order_desc=desc",
       "https://iss.moex.com//iss/statistics/engines/stock/markets/shares/correlations",
       "https://iss.moex.com/iss/statistics/engines/stock/deviationcoeffs",
       "https://iss.moex.com//iss/engines/stock/markets/shares/securities/YNDX/candles?from=2019-11-01&interval=60&start=100",
       "https://iss.moex.com/iss/rms/engines/stock/objects/irr",
       "https://iss.moex.com/iss/statistics/engines/stock/quotedsecurities",
       "https://iss.moex.com/iss/engines/stock/markets/shares/trades.xml?securities=SBER"]

def parse_micex(url:str):
    """
    parse data from MICEX API and converts it to DataFrame
    """

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    #parse columns names and their types
    cols_ = [x['name'].lower() for x in soup.find_all('columns')[0].find_all('column')]
    col_types = {x['name']:x['type'] for x in soup.find_all('column')}

    try:
        #collects meta data for future loop : while index < total collect data
        meta_columns = [x['name'].lower() for x in soup.find_all('columns')[1].find_all('column')]
        meta = {k:int(x[k]) for k in meta_columns for x in soup.find_all('rows')[1].find_all('row')}
    except Exception as e:
        print(str(e))

    #create DataFrame
    df = pd.DataFrame({i:{k:x[k] for k in cols_} for i, x in enumerate(soup.find_all('rows')[0].find_all('row'))}).T

    return df

def main():
    print(parse_micex(url[0]))

if __name__ == '__main__':
    main()
