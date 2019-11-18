import numpy as np
import pandas as pd
import pandas_datareader.data as web

import datetime

from sklearn.linear_model import LinearRegression

start_date = datetime.datetime(2014,6,4)
end_date=datetime.datetime(2019,11,18)

model = LinearRegression()

micex = web.DataReader('IMOEX.ME',  'yahoo', start=start_date,end=end_date)
gazp = web.DataReader('GAZP.ME','yahoo',start=start_date,end=end_date)
ya = web.DataReader('YNDX.ME','yahoo',start=start_date,end=end_date)
sber=web.DataReader('SBER.ME','yahoo',start=start_date,end=end_date)

df = pd.concat([micex[['Adj Close']],ya[['Adj Close']],gazp[['Adj Close']],sber[['Adj Close']]],axis=1).dropna()

df.columns=['IMOEX','YA','GAZP','SBER']

df.tail()


print('# Correlation coefficients')
print(df.corr())


X=df[['YA','GAZP','SBER']]
Y=df['IMOEX']

model.fit(X.iloc[:-10],Y.iloc[:-10])

r_sq = model.score(X,Y)

print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

y_pred = model.predict(X.iloc[-10:])
print('predicted response:', y_pred, sep='\n')
