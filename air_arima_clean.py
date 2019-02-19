import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import required library

def parse(x):
#function for parsing data into required format
  return datetime.strptime(x, '%Y %m %d %H')

def preprocess_data(inputFile,outputFile):
#basic preprocessing for converting raw inputfile to preprocessed file
#for changing dataset this function need to be updated
  dataset = pd.read_csv(inputFile,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
  dataset.drop('No', axis=1, inplace=True)
  # manually specify column names
  dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
  dataset.index.name = 'date'
  # mark all NA values with 0
  dataset['pollution'].fillna(0, inplace=True)
  # drop the first 24 hours as it has 0 value
  dataset = dataset[24:]
  # summarize first 5 rows
  print(dataset.head(5))
  # save to file
  dataset.to_csv(outputFile)
preprocess_data('raw.csv','pollution.csv')

#read preprocessed csv file
df=pd.read_csv('pollution.csv',header=0, index_col=0)
print(df)
#plot various parameters of csv into graph
df[['dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']].plot()
plt.show()

#make dataset indexed with time
df.index = pd.DatetimeIndex(df.index)

#we will predict pollution using pressure so analyse its distribution
#change press to another column to train model for different column
print(sm.tsa.stattools.adfuller(df['press']))
print(sm.tsa.stattools.adfuller(df['pollution']))

#get difference between each rows in pressure column as lag
df['lag']=df['press'].shift()
#remove null values
df.dropna(inplace=True)
#Train arima to predict pollution
model3=sm.tsa.ARIMA(endog=df['pollution'],exog=df[['lag']],order=[1,1,0])
results3=model3.fit()
#get various statstics related to training
print(results3.summary())
print(model3.predict(df[['lag']]))

