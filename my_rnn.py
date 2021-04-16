# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.get_data import get_data
from utils.plots import plt_long_series
'''
goals:
is there relation between historical data of tsetmc and price? 
1. train it with data that related to a one share.
2. combine all shares data and train it.
3. 

plan:

1. create class 1 or 0, 1 --> next day price increase / 0 --> deacrese

1.1. simple NN only for price 
1.2. simpleRNN only for price, add extra features one by one and trace changes.

'''
# %%
data_eg = get_data('وبملت')
    
# %%
data= data_eg[['adjClose']].values


# %%
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
# %%
