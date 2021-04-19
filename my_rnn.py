# %%
from typing import Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.get_data import get_InCo_df, get_price_df
from utils.plots import plt_long_series
from utils.pre_processing import priceTransformer
from sklearn.model_selection import train_test_split

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
def get_data(share_name):
    df_inco = get_InCo_df(share_name)
    df_price = get_price_df(share_name)

    df_data = df_price.join(df_inco, how='inner')

    return df_data
# %%
def get_tr_price_df(share_name):
    price_df = get_price_df(share_name)
    p_tr = priceTransformer(drop=True, drop_pre_day_adj=True)
    price_df_tr = p_tr.transform(price_df)
    return price_df_tr

# %%
import os
SHARE_FOLDER = 'data/tickers_data'
share_name_list = [share_name_dot_csv.replace('.csv', '') \
                    for share_name_dot_csv in os.listdir(SHARE_FOLDER)]  
# %%
def window(data :pd.DataFrame, size) -> pd.DataFrame:
    df = pd.concat([data.shift(-i) for i in range(size)], axis=1, 
            keys =[f't{i}' for i in range(size)])
    df.dropna(inplace=True)
    return df
# %%
def up_down(x):
    if x>0:
        return True
    if x<=0:
        return False
def get_y(w_data, n_day):
    return (w_data[n_day]['adjClose_per'].apply(up_down)).astype(int)
    
def get_xy_df(tr_data:pd.DataFrame, window_size:int):
    w_data = window(tr_data, window_size)
    
    X = w_data.loc[:,'t0':f't{window_size-2}']
    # y = (w_data.loc[:,f't{window_size-1}']['adjClose_per'].apply(up_down)).astype(int)
    last_time_step = f't{window_size-1}'
    y = get_y(w_data, last_time_step)

    return X,y

# %%


tr_price_df = get_tr_price_df('وبملت')
X, y = get_xy_df(tr_price_df, 15)

y13 = get_y(X,'t13')
cm_baseline_metric = metrics.confusion_matrix(y, y13)
score = (cm_baseline_metric[0,0] + cm_baseline_metric[1,1])/sum(cm_baseline_metric.reshape(-1))
print(f'score of naive model is :{score}')

# %%

# tr_price_df = get_tr_price_df('وبملت')
# X, y = get_xy_df(tr_price_df, 15)
# X13, y14 = get_xy_df(tr_price_df, 14)

# X_train, X_test , y_train, y_test = train_test_split(X, y)

# %%
# def prepair_y(share_name,time_n):
#     price_df = get_price_df(share_name)
#     price_df_w =  window(price_df['adjClose'],time_n)
#     n_colums = price_df_w.shape[1]
#     rezult = (price_df_w.iloc[:,n_colums-2]<price_df_w.iloc[:, n_colums-1]).astype(int)
    
#     return rezult
# share_name = 'وبملت'
# y14 = prepair_y(share_name,14)
# # ii index_intersection
# ii = y14.index.intersection(y.index)
# cm_baseline_metric = metrics.confusion_matrix(y.loc[ii], y14[ii])
# score = (cm_baseline_metric[0,0] + cm_baseline_metric[1,1])/sum(cm_baseline_metric.reshape(-1))
# print(f'score of naive model is :{score}')

# %%


# %%
# y14.drop(y14.tail(1).index, inplace=True)
# # why i need drop? window with size n create Nan rows in
# # last n row, so itself must drop it. 
# # instead of above code we can use :
# # ii = y14.index.intersection(y.index)

# cm_baseline_metric = metrics.confusion_matrix(y, y14)
# score = (cm_baseline_metric[0,0] + cm_baseline_metric[1,1])/sum(cm_baseline_metric.reshape(-1))
# print(f'score of naive model is :{score}')
# %%
def get_all_xy_df(share_name_list, window_size):
    X_tmp = []
    y_tmp = []

    for share_name in share_name_list:
        tr_price_df = get_tr_price_df(share_name)
        X,y = get_xy_df(tr_price_df, window_size)

        X_tmp.append(X)
        y_tmp.append(y)

    X_all = pd.concat(X_tmp, ignore_index=True)
    y_all = pd.concat(y_tmp, ignore_index=True)
    return X_all, y_all


X_all, y_all = get_all_xy_df(share_name_list, 15)
X_train, X_test , y_train, y_test = train_test_split(X_all, y_all)

# %%
#calculate score of baseline metric for dataset of all shares info.
y13 = get_y(X_all,'t13')
cm_baseline_metric = metrics.confusion_matrix(y_all, y13)
score = (cm_baseline_metric[0,0] + cm_baseline_metric[1,1])/sum(cm_baseline_metric.reshape(-1))
print(f'score of naive model is :{score}')
# %%


#1. implementing a logistic regression.
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

predictions = log_reg.predict(X_test)
score = log_reg.score(X_test, y_test)
print(f'score is: {score}')

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)
print(f'confusion matrix is: \n {cm}')
# %%

# %%
#2. implementing a simple NN

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape= X_train.shape[1:]),
    tf.keras.layers.Dense(1)
])
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# %%
model.fit(X_train, y_train,validation_split=0.20, epochs=300)


# %%
#3. implementing a RNN

def convert_df2np(w_df1:pd.DataFrame) -> np.ndarray:
    '''
    e.g.
    l1 = np.arange(12).reshape(6,2)
    df1 = pd.DataFrame(l1)
    w_df1 = window(df1, 3)
    convert_df2np(w_df1)
    '''
    # Todo : clean up!
    # after sclicing X.columns.levels[0] give us base DataFrame levels 
    # for more info see https://stackoverflow.com/questions/28772494/how-do-you-update-the-levels-of-a-pandas-multiindex-after-slicing-its-dataframe
    # so i replace it with 
    # w_df1.columns.get_level_values(0).unique()
    # and replace w_df1.columns.levshape[0] with len(level0)
    level0 = list(w_df1.columns.get_level_values(0).unique())

    n1 = np.zeros(shape=(w_df1.shape[0], 
                        len(level0),
                        w_df1.columns.levshape[1]))
    
    for i in range(len(level0)):
        n1[:, i, :] = w_df1.loc[:,level0[i]]
    return n1.astype(np.float32)

# %%
#sequence to vector data
from tensorflow import keras
X_np = convert_df2np(X)
y_np = y.to_frame().values

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 5]),
    keras.layers.SimpleRNN(10),
    keras.layers.Dense(1)
])
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(X_np, y_np,validation_split=0.20, epochs=250)
# %%

# create a sequence to sequence RNN 
# train without related data to big up and down that happend
# is that predictable ? 