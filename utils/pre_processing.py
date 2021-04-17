import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class priceTransformer(BaseEstimator, TransformerMixin):
    '''
    a transformer for price dataframe
    add previous days adjClose price to each day
    and add change percentage of open high low adjclose
    to dataframe.
    and remove theme.

    '''
    def __init__(self, price_ch_per=True, drop_pre_day_adj=False,drop=False):
        self.price_ch_per = price_ch_per
        self.drop_pre_day_adj = drop_pre_day_adj
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.price_ch_per :
            X['pre_day_adjClose'] =  X['adjClose'].shift(1)
            X.dropna(inplace=True)
            X['open_per'] = (X['open'] - X['pre_day_adjClose']) * 100 / X['pre_day_adjClose']
            X['high_per'] = (X['high'] - X['pre_day_adjClose']) * 100 / X['pre_day_adjClose']
            X['low_per'] = (X['low'] - X['pre_day_adjClose']) * 100 / X['pre_day_adjClose']
            X['adjClose_per'] = (X['adjClose'] - X['pre_day_adjClose']) * 100 / X['pre_day_adjClose']
            X['close_per'] = (X['close'] - X['pre_day_adjClose']) * 100 / X['pre_day_adjClose']

            X.drop(['close','open','high','low','adjClose'], inplace=True, axis=1)
            if self.drop_pre_day_adj:
                X.drop(['pre_day_adjClose'], inplace=True, axis=1)
            if self.drop :
                X.drop(['value','volume','count'], inplace=True, axis=1)


        
        return X
