# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.get_data import get_InCo_df, get_price_df
from utils.plots import plt_long_series
from utils.pre_processing import priceTransformer
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
data_eg = get_price_df('وبملت')
p_tr = priceTransformer(drop=True, drop_pre_day_adj=True)
data_tr = p_tr.transform(data_eg)
data_tr
# %%
data= data_tr

def window(data :pd.DataFrame, size) -> pd.DataFrame:
    df = pd.concat([data.shift(-i) for i in range(size)], axis=1, 
            keys =[f't{i}' for i in range(size)])
    df.dropna(inplace=True)
    return df

window_size = 10

w_data = window(data, window_size)
X = w_data.loc[:,'t0':f't{window_size-2}']
y = (w_data.loc[:,f't{window_size-1}']['adjClose_per'] > w_data.loc[:,f't{window_size-2}']['adjClose_per']).astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y)

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
#2. implementing a simple NN

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape= X_train.shape[1:]),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
# %%
model.fit(X_train, y_train, epochs=200)


# %%
#3. implementing a RNN

# %%
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

model.fit(X_np, y_np, epochs=250)
# %%
