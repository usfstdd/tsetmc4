# tsetmc4

# analyse iran stock makert with NNs.

i have data of individual and corporate historical activities and price historical data of all shares in data/client_types_data and data/tickers_data.

## 1. analyse candlestick patterns in iran stock market.
analyse historical price changes with data of (high, low, open, close, adjClose) that i have for each day.

### 1.1 preprocessing approach:
* i preprocess these to percentage change with recpect to  previous day's adjClose.
this give me these privileges:
1. no need to standards the data.
2. combine all shares data to have rich dataset of candlestick overtime. so i can study candlestick patterns.

* with window function i can create instance that have N days of price information. for example if the days sort is 0-1-2-3-4-5-6 and window_size is 3 i have 0-1-2, 1-2-3, 2-3-4, 3-4-5, 4-5-6 as instances. in this README file default window size is 15.


### 1.2 is there relation between previous days data and future price?

to answer the question,i reduced the regression problem(what is tomorow price?) to classification (trend prediction or tomotow price goes up or down?)

to prevent fooled by NNs, i create a simple model that returns last days trend as predicted value. (so NNs that will create must can predict trend direction changes to overcome the score of simple model.)

share_name = وبملت 
score of simple model for share_name: 0.63
score of simple model for all shares :0.6773660608921407

