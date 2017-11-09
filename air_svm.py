import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sklearn
#import required library
def parse(x):
#function for parsing data into required format
  return datetime.strptime(x, '%Y %m %d %H')

def preprocess_data(inputFile,outputFile):
#basic preprocessing for converting raw inputfile to preprocessed file
#for changing dataset this function need to be updated
  dataset = pd.read_csv(inputFile)
  #drop date part because svm can not utilize time information
  dataset.drop('year', axis=1, inplace=True)
  dataset.drop('month', axis=1, inplace=True)
  dataset.drop('day', axis=1, inplace=True)
  dataset.drop('hour', axis=1, inplace=True)
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

#Store training dataset in X
X=df.values
#convert wind direction from string to numerical value
encoder = LabelEncoder()
X[:,4] = encoder.fit_transform(X[:,4])
X = X.astype('float32')
#remove polution colunm from data into label
#Split X
y=X[:,0]
X=X[:,1:]

#create SVM model and train it
clf = svm.SVC(kernel = 'linear', C = 1.0)
# ensure all data is float
y = (y > 60).astype(float)
clf.fit(X, y)

#predict values
y_pred=clf.predict(X)
print("Actual Y:")
print(y)
print("Predicted Y:")
print(y_pred)
a=sklearn.metrics.accuracy_score(y, y_pred, normalize=True, sample_weight=None)
print("accuracy",a)
