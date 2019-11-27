import pandas as pd
import requests

from bs4 import BeautifulSoup

url = ["https://iss.moex.com//iss/statistics/engines/stock/markets/shares/correlations",
       "https://iss.moex.com/iss/statistics/engines/stock/deviationcoeffs",
       "https://iss.moex.com//iss/engines/stock/markets/shares/securities/YNDX/candles?from=2019-11-01&interval=60&start=100",
       "https://iss.moex.com/iss/rms/engines/stock/objects/irr",
       "https://iss.moex.com/iss/statistics/engines/stock/quotedsecurities",
       "https://iss.moex.com/iss/engines/stock/markets/shares/trades.xml?securities=SBER"]

r = requests.get(url[0])

soup = BeautifulSoup(r.text, 'lxml')

cols_ = [x['name'].lower() for x in soup.find_all('columns')[0].find_all('column')]
try: 
    meta_ = [x['name'].lower() for x in soup.find_all('columns')[1].find_all('column')]
except Exception as e:
    print(str(e))
col_types = {x['name']:x['type'] for x in soup.find_all('column')}

df = pd.DataFrame(columns=cols_)

for d in [{k:x[k] for k in cols_} for x in soup.find_all('rows')[0].find_all('row')]:
    df = df.append(d,ignore_index=True)
    
