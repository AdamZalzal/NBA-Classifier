#This is an NBA position predictor based player stats every season since approximately 1950
#The code is adapted from Google's Machine Learning Crash Course which I completed prior to this project.


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

#Reading in data from CSV
stats = pd.read_csv('nba_stats_for_analysis.csv', skip_blank_lines = True, usecols = ['Pos','G','MP','ORB','DRB','TRB','AST','STL','BLK','TOV','PTS','3PA','3P','FGpct','FG','FGA','2P','2PA','2Ppct','3Ppct','FT','FTA','FTpct','PER'])
stats = stats.reindex(np.random.permutation(stats.index)) #shuffle data

#Creating custom features
stats['ppg'] = (stats['PTS']/stats['G'])
stats['rpg'] = (stats['TRB']/stats['G'])
stats['apg'] = (stats['AST']/stats['G'])

#Dropping NaN values
stats = stats.dropna()

#Plotting averages on box and whisker plots to identify outliers
plt.boxplot(stats['ppg'])
plt.show()
plt.boxplot(stats['rpg'])
plt.show()
plt.boxplot(stats['apg'])
plt.show()

#Removing outliers
stats = stats[(stats['ppg']<25.15) & (stats['rpg']<9.37) & (stats['apg']< 6.03)]

#Plotting Correlation Map
f = plt.figure(figsize=(19, 15))
plt.matshow(stats.corr(), fignum=f.number)
plt.xticks(range(stats.shape[1]), stats.columns, fontsize=14, rotation=45)
plt.yticks(range(stats.shape[1]), stats.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

#Extracting Player Postion (labels)
position = stats['Pos']
del stats['Pos']


#Binary encoding labels to form one hot vectors
lb = LabelBinarizer()
lb.fit(position)
encoded_positions = lb.transform(position)

#Standardizing Training Data
stats = stats.astype(float)
scaler = StandardScaler()
scaler.fit(stats)
scaled_stats = scaler.transform(stats)

#Splitting data in train and test sets
X_train, x_test, Y_train, y_test = train_test_split(scaled_stats, encoded_positions, test_size = 0.1, random_state = 42)

#Initializing model and specifying topology
model = Sequential()
input_size = len(stats.columns.values)
model.add(Dense(20,input_dim = input_size, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'sigmoid'))

#Defining optimizer
opt = Adam(lr = 0.0001)

#Compiling the model
model.compile(optimizer = opt,
            loss = 'categorical_crossentropy', 
            metrics = ['acc'])

#Training the model using 20% of data as validation set
history = model.fit(
    X_train,
    Y_train,
    batch_size= 10,
    epochs = 50,
    validation_split = 0.2,
)

#Plot Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Possible Next Steps:
    #Grid Search to identfiy optimal hyperparameters. 
    #K-Folds Cross Validation for testing.



