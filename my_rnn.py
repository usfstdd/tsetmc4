# %%
from datetime import time
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
0. start with high, low, close, open, adjclose
1. train it with data that related to a one share.
2. combine all shares data and train it.
3. 

plan:

1. to see relation between price and historical data, 
i predict class 0 when price goes down and class 1 when price goes up.

1.1. simple NN only for price 
1.2. simpleRNN only for price, add extra features one by one and trace changes.

'''

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
def getXY(share_name, X_steps, y_steps, predict_at_last_step=True):
    window_size = X_steps + y_steps
    data = window(get_tr_price_df(share_name), window_size)


    time_steps = list(data.columns.get_level_values(0).unique())
    features = list(data.columns.get_level_values(1).unique())
    m=data.shape[0]

    data_np =np.empty((m, len(time_steps), len(features)))
    for t in range(len(time_steps)):
        data_np[:,t,:] = data[time_steps[t]].values

    if predict_at_last_step:
        X_np = data_np[:,:X_steps,:]
        y_np = (data_np[:, X_steps:, 3]>0).astype(np.int32)
        return data, X_np, y_np

    if not predict_at_last_step:
        X_np = data_np[:, :X_steps ,:]
        y_np = np.empty((m, X_steps, y_steps))
        for t in range(1, y_steps+1):
            y_np[:, :, t-1] = data_np[:, t:t+X_steps, 3]
            y_np = (y_np>0).astype(np.int32)

        return data, X_np, y_np

        
# %%

# %%

from sklearn import metrics
# tr_price_df = get_tr_price_df('وبملت')
# X, y = get_xy_df(tr_price_df, 15)

# y13 = get_y(X,'t13')
share_name = 'وبملت'
data, X, y = getXY(share_name, 14, 1)
y14 = (data['t13']['adjClose_per'] > 0).astype(np.int32)

cm_baseline_metric = metrics.confusion_matrix(y, y14)
score = (cm_baseline_metric[0,0] + cm_baseline_metric[1,1])/sum(cm_baseline_metric.reshape(-1))
print(f'score of naive model is :{score}')
X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%
def get_all_xy_df(share_name_list):
    data_tmp = []
    X_tmp = []
    y_tmp = []

    for share_name in share_name_list:
        
        data, X,y = getXY(share_name, 14,1)

        X_tmp.append(X)
        y_tmp.append(y)
        data_tmp.append(data)

    X_all = np.concatenate(X_tmp, axis=0)
    y_all = np.concatenate(y_tmp, axis=0)
    data_all =pd.concat(data_tmp, ignore_index=True)
    return data_all, X_all, y_all


data_all, X_all, y_all = get_all_xy_df(share_name_list)
X_train, X_test , y_train, y_test = train_test_split(X_all, y_all)

# %%
#calculate score of baseline metric for dataset of all shares info.
y14 = (data_all['t13']['adjClose_per'] > 0).astype(np.int32)
cm_baseline_metric = metrics.confusion_matrix(y_all, y14)
score = (cm_baseline_metric[0,0] + cm_baseline_metric[1,1])/sum(cm_baseline_metric.reshape(-1))
print(f'score of naive model is :{score}')
# %%


#1. implementing a logistic regression.
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train.reshape(-1, 14 * 5), y_train)

predictions = log_reg.predict(X_test.reshape(-1, 14 * 5))
score = log_reg.score(X_test.reshape(-1, 14 * 5), y_test)
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

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()
# %%
checkpoint_path = 'training_1/simpleNN.ckpt'
os.path.dirname(checkpoint_path)
simple_NN_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
# %%
history = model.fit(X_train,
          y_train,
          validation_split=0.20, 
          epochs=30,
          callbacks=[simple_NN_callback])

# %%
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