from __future__ import print_function
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import LSTM
# import various required library for
# any relevant needs here.

# This is git trial change

def this_is_dummy_function(none):
	return

def parse(x):
    # function for parsing data into required format
    return datetime.strptime(x, '%Y %m %d %H')


def preprocess_data(inputFile, outputFile):
    # basic preprocessing for converting raw inputfile to preprocessed file
    # for changing dataset this function need to be updated
    dataset = read_csv(inputFile, parse_dates=[
                       ['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp',
                       'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours as it has 0 value
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv(outputFile)

# This is a comment - important/.


def visualize(inputFile, groups):
    # plot columns definded in "groups" from inputFile
    dataset = read_csv(inputFile, header=0, index_col=0)
    values = dataset.values

    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()


# call function for preprocessing and visualizing data
preprocess_data('raw.csv', 'pollution.csv')
visualize('pollution.csv', [0, 1, 2, 3, 5, 6, 7])


def convert_timeseries(data, n_in=1, n_out=1, dropnan=True):
    # covert timeseries data to t-n to t-1 form
    # n defines how many previous value should be taken into consideration
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


 #-----------------------------------#
 #		Actual code starts here   	 #
 #-----------------------------------#
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# encode direction into integer
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = convert_timeseries(scaled, 1, 1)

print(reframed.head())
# drop columns we don't want to predict
# need to change this if we change N or change dataset
reframed.drop(
    reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24  # 1 year
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X = train[:, :-1]
train_y = train[:, -1]
test_X = test[:, :-1]
test_y = test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='Training Loss')
pyplot.plot(history.history['val_loss'], label='Validation Loss')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast to revert data into original form
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
# Actial Input
inv_xp = inv_yhat[:, 1:]
# predicted output
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
# Actual output
inv_y = inv_y[:, 0]
print("Actial Input:")
print(inv_xp)
print("Actual output:")
print(inv_y)
# predicted output will be offset by 1
print("Predicted output:")
print(inv_yhat)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
