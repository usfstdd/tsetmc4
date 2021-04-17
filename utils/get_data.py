import pandas as pd
import numpy as np

def get_InCo_df(share_name):
    #In stands for Individual
    #Co stands for corporate
    DIR = 'data/client_types_data'
    df =  pd.read_csv(f'{DIR}/{share_name}.csv', index_col='date')
    df.dropna(inplace=True)

    return df


def get_price_df(share_name):
        df =  pd.read_csv(f'data/tickers_data/{share_name}.csv', index_col='date')

        return df





'''
>>>data.values
array([[1050.],
       [1065.],
       [1061.],
       ...,
       [4170.],
       [4210.],
       [4310.]])
'''






