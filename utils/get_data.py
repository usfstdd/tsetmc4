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

def get_data(share_name):
    df_inco = get_InCo_df(share_name)
    df_price = get_price_df(share_name)

    df_data = df_price.join(df_inco, how='inner')

    return df_data



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

def window(data, window_len):
    '''
    convert input data with
    shape (sample_count, features) to 
    shape (:, time_step, n_features)
    '''

    m = len(data)-(window_len-1)
    n_features = len(data[0])

    X = np.zeros(shape=(m,window_len,n_features))

    for i in range(window_len):
            X[:,i,:] = data[i:i+m,:]

    return X
# prepaire y for a sequence to sequence model :




def windowXY(data, n_time_steps, n_steps_ahead, target_feature=0):
    
    print(f'data is:\n{data}')
    X = window(data,n_time_steps)
    print(f'X is:\n{X}')
    


    m = len(X) - (n_steps_ahead)
    Y = np.zeros(shape=(m, n_time_steps, n_steps_ahead))
    
    for j in range(n_steps_ahead):
        for i in range(n_time_steps):
            Y[:,i, j] = data[(i+j+1):(i+j+1)+m, target_feature]

    return X[:len(Y)], Y