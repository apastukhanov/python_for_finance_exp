import requests
import pandas as pd

import datetime
from bs4 import BeautifulSoup

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

from sklearn.svm import SVC


urls = {
    'gcurve' : 'http://www.cbr.ru/hd_base/zcyc_params/?FromDate={}',
    'mosprime' : 'https://www.cbr.ru/hd_base/mosprime/?FromDate={}',
    'ruonia':'https://www.cbr.ru/hd_base/ruonia/?FromDate={}',
    'deposits': 'https://www.cbr.ru/vfs/statistics/pdko/int_rat/deposits_30.xlsx',
}

def make_data_frame(rate_type, date = "01.01.2014"):
    
    r = requests.get(urls[rate_type].format(date))

    soup = BeautifulSoup(r.text,'lxml')

    cols = [[y.text.replace('\r\n','') for y in x.find_all('th')] for x in soup.find_all('tr') if [y.text for y in x.find_all('th')]!=[]]
    data = [[y.text for y in x.find_all('td')] for x in soup.find_all('tr') if [y.text for y in x.find_all('td')]!=[]]

    df = pd.DataFrame(data)

    if rate_type == 'gcurve':
        cols = ['Дата']+cols[1]        
        df.columns = cols

    else:
        df.columns = cols[0]
    
    return df


def collect_dfs():
    for k in urls.keys():
        if k == 'deposits':
            continue
        df = make_data_frame(k)
        with pd.ExcelWriter(k+'.xlsx','xlsxwriter') as f:
            df.to_excel(f)
   
#read data
mosprime = pd.read_excel("mosprime.xlsx", index_col=0)
gcurve = pd.read_excel("gcurve.xlsx", index_col=0)
ruonia = pd.read_excel("ruonia.xlsx", index_col=0)
deposits = pd.read_excel("deposits_30.xlsx", index_col=0, sheet_name='ставки_руб.')

dfs = [mosprime,gcurve,ruonia,deposits]

 MONTHS = [
        'Январь',
        'Февраль',
        'Март',
        'Апрель',
        'Май',
        'Июнь',
        'Июль',
        'Август',
        'Сентябрь',
        'Октябрь',
        'Ноябрь',
        'Декабрь'
    ]


def make_mcalendar(MONTHS):
    months_dict = {k:v for k,v in zip(range(1,13),MONTHS)}
    return months_dict
    
def preprocc_df(df):
    
    df = df.drop(df.loc[df[df.columns[1]]==' — '].index)

    df['Дата'] = [datetime.datetime.strptime(d,"%d.%m.%Y") for d in df['Дата']]

    for col in df.columns[1:]:
        df[col]=df[col].str.replace(',','.').astype('float32')

    df['Дата'] = df['Дата'].astype('datetime64')

    df['Месяц'] = df['Дата'].dt.month
    df['Год'] =  df['Дата'].dt.year

    months_dict = make_mcalendar(MONTHS)

    df['Месяц_Год'] = [months_dict[m]+' '+str(y) for m,y in zip(df['Месяц'], df['Год'])]
    
    return df
   
df2 =  preprocc_df(dfs[1])

df2 = df2.groupby('Месяц_Год').mean()


def clean_dep30(df1):
    cols = df1.iloc[3,12:-1].values
    inds = df1.iloc[4:-1,0]

    df1 = df1.iloc[4:-1,12:-1]
    df1.columns = cols
    
    df1 = df1.reset_index()
    
    df1.rename({df1.columns[0]:'Месяц_Год'}, axis = 1, inplace=True)
    
    df1 = df1.set_index('Месяц_Год')
    
    return df1
    

df1 = clean_dep30(dfs[3])


target_col_x = '0,25'
target_col_y = 'от 31 до 90 дней'

dataset = pd.concat([df1[[target_col_y]],df2[[target_col_x]]], axis=1,sort=False)

sample = dataset.loc[dataset[dataset.columns[0]].isna()==False].copy()
pred = dataset.loc[dataset[dataset.columns[0]].isna()].copy()

sample[target_col_y]=sample[target_col_y].astype('float32')

sample.corr()

plt.scatter(sample[target_col_y],sample[target_col_x])
plt.show()

model = LinearRegression()
# model = SVC(C=1.0, kernel='linear', gamma='auto')

samp_size = 10

X_train = np.array(sample[target_col_x]).reshape(-1, 1) #[:-samp_size]
y_train = np.array(sample[target_col_y])#[:-samp_size]

X_test = np.array(sample[target_col_x]).reshape(-1, 1)#[-samp_size:]
y_test = np.array(sample[target_col_y])#[-samp_size:]

model.fit(X_train,y_train)

print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

print()

y_pred = model.predict(X_test)
print('predicted response:', y_pred, sep='\n')

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter(np.array(sample[target_col_x]),np.array(sample[target_col_y]))
plt.plot(X_test,y_pred,c='red')
plt.show()

model.predict(np.array(pred[target_col_x]).reshape(-1, 1))
