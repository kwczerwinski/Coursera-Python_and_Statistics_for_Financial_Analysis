import pandas as pd
# if plot is not visible (probably a default behaviour PyCharm)
from matplotlib import pyplot as plt

# import data from file
data = pd.read_csv('data/TRADEGATE%DGD=D.csv', sep='\t', index_col=0,
                   names=('timestamp', 'open', 'high', 'low', 'close', 'volume', 'is_generated'))

# show first 5 rows
print(data.head())

# show all indexes
print(data.index)

# show first index
print(data.index[0])

# show last index
print(data.index[-1])

# show columns (index column is omitted)
print(data.columns)

# show number of rows and columns (index column is omitted)
print(data.shape)

# show last 5 rows
print(data.tail())

# show basic statistics for each column
print(data.describe())

# slicing DataFrame
# slicing by label - .loc[<index_name>, '<column_name']
print(data.loc[1716940800.0, 'high'])
# slicing by position - .iloc[<index_position>, <column_position>] positions are numbered from 0
print(data.iloc[0, 0])
# slicing with multiple positions
print(data.loc[1716768000.0:1716940800.0, 'open'])  # first and last position included
print(data.iloc[0, :])  # all columns
print(data.loc[1716940800.0, ['open', 'close']])  # specified columns

# filtering rows
print(data[data['open'] > data['close']])  # show only bearish bars

# build-in plot in DataFrame (module matplotlib must be installed)
# data.iloc[:9, 3].plot()  # plot first 10 close prices
# data.iloc[10:19, 3].plot()  # plot next 10 close prices
# data.iloc[20:29, 3].plot()  # there will be a gap between each lines

# get specified column
print(data['close'])  # column has no name
print(data.loc[:, ['close']])  # column has name

# get multiple columns
print(data[['open', 'close']])  # columns has names

# create new column
data['next_close'] = data['close'].shift(-1)  # new column with close price of next row
# creating new column is saved in data
print(data.loc[:, ['close', 'next_close']].head())

# shift column upward one row
print(data['open'].shift(-1).head())  # shifting is not permanent

# create new column with close price difference between actual and next close prices
data['price_diff'] = data['next_close'] - data['close']
print(data.loc[:, ['close', 'next_close', 'price_diff']].head())

# column with return for each timestamp
data['return'] = data['price_diff'] / data['close']
print(data.loc[:, ['close', 'price_diff', 'return']].head())

# column with direction for each timestamp
data['direction'] = [1 if pd > 0 else -1 if pd < 0 else 0 for pd in data['price_diff']]
# 1 if price go up
# -1 if price go down
# 0 if no change in price
# code is different from lesson, but I think it is a little bit better
print(data.loc[:, ['price_diff', 'direction']].head())

# column with average of 3 last close prices
data['avg_3'] = (data['close'] + data['close'].shift(1) + data['close'].shift(2)) / 3
print(data.loc[:, ['close', 'avg_3']].head())
# first two rows will have NaN (Not a Number) in new column because there are no numbers before first timestamp

# columns wiht moving averages
data['MA40'] = data['close'].rolling(40).mean()
data['MA200'] = data['close'].rolling(200).mean()
print(data.loc[:, ['close', 'MA40', 'MA200']].tail())

# plot close price, MA40 and MA200
# data['close'].tail(1000).plot()
# data['MA40'].tail(1000).plot()
# data['MA200'].tail(1000).plot()

# Simple strategy using Moving Averages
data['MA10'] = data['close'].rolling(10).mean()  # fast signal
data['MA50'] = data['close'].rolling(50).mean()  # slow signal
# data['close'].tail(1000).plot()
# data['MA10'].tail(1000).plot()
# data['MA50'].tail(1000).plot()

# buy 1 share if MA10>MA50 else sell it
data['shares'] = [1 if data.loc[i, 'MA10'] > data.loc[i, 'MA50'] else 0 for i in data.index]
# profit from using strategy above (simple calculations since we can only have 0 or 1 share)
data['shares_profit'] = [data.loc[i, 'price_diff'] if data.loc[i, 'shares'] > 0 else 0 for i in data.index]
print(data[data['shares'] > 0].loc[:, ['shares_profit']].head(50))
# data['shares_profit'].plot()
# data[data['shares'] > 0]['shares_profit'].plot()

# calculating wealth
data['wealth'] = data['shares_profit'].cumsum()
print(data[['price_diff', 'shares', 'shares_profit', 'wealth']].tail(10))
print(data.columns)
# data['wealth'].plot()

data['money_spent'] = [data.loc[i, 'close'] if data.loc[i, 'shares'] > 0 and data.shift(1).loc[i, 'shares'] == 0 else 0 for i in data.index]
data['money_spent_sum'] = data['money_spent'].cumsum()
print(data[['close', 'shares', 'money_spent', 'money_spent_sum']].tail(100).head(50))

data['money_gain'] = [data.loc[i, 'close'] if data.loc[i, 'shares'] == 0 and data.shift(1).loc[i, 'shares'] > 0 else 0 for i in data.index]
data['money_gain_sum'] = data['money_gain'].cumsum()
print(data[['close', 'shares', 'money_gain', 'money_gain_sum']].tail(50))

print('Total money spent: ', data['money_spent_sum'].dropna().iloc[-1])
print('Total money gain: ', data['money_gain_sum'].dropna().iloc[-1])
print('Total profit: ', data['wealth'].dropna().iloc[-1])

plt.show()
# plot must be closed for program to continue (don't know if it's default behaviour in python)

pass
