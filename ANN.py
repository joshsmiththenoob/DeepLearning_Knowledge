# predict which person has highest risk to leave the bank from correlations between these features
# to prevent the "max-guests" from leaving back.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
# from keras import Sequential


print(tf.__version__)           # Test if tensorflow has installed

# Part I - Data Preprocessing

# Import dataset 
# There's 2-D DataFrame
Data_Table = pd.read_csv(r'./Churn_Modelling.csv')
X = Data_Table.iloc[:, 3:-1].values             # got table of features -> 2D NumpyArray -> decide to get which columns are features relevant to label
Y = Data_Table.iloc[:, -1]                      # got coressponding label -> 1D Series -> dependent variable on indepedent variable x (features)


# Encode categorical data
# Label Encode the "Gender column" ; Gender column -> X[:,2]
# print(X[:, 2])
le = LabelEncoder()
# Label Encode - fit_transform : turn the text feature into a digi number feature like "Female -> 0, Male -> 1"
X[:, 2] = le.fit_transform(X[:, 2])

# print(X[:,2])

# One-Hot Label Encode the "Geography" column -> X[:,1]
# Label Encode vs One-Hot Label Encode : One-Hot Label Encode does NOT have order and scale relationshiop between France,Spain and Germany
# use [1] in ColumnTransformer to One-Hot Label Encode -> X[:,1]
# print(X,X.shape)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough' )
X = np.array(ct.fit_transform(X))
# the transformed column(s) become the leftmost columns to get new table
print(X,X.shape)
print(type(X[0,8]))


# Split the dataset into the Training set and Test set
# train_data : test_data = 0.8: 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)

# Feature Scaling (Compulsory) : let all data of features be One-Scale so that Computer will get less computational burden 
# Ex : set all dataset to be Normalized -> The feature dataset ranges between 0 and 1.
# Scale all feature dataset
# Standardized the feature dataset -> scaled feature data = x1(origin data) - U (mean) / O (Standard Deviation)
sc = StandardScaler()
# fit_transoform for train (known data) and transform for test (unknown data) so that model can recognize the known (train) data and unknown (test) data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print('Training data after Scaling :',X_train)
# print('Testing data after Scaling :',X_test)

# Part II - Building the ANN

# Initializing the ANN
# Create Object
ann = tf.keras.models.Sequential()
# Add input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units = 11, activation= 'relu'))
# Add the second hidden layer
ann.add(tf.keras.layers.Dense(units = 11, activation= 'relu'))
# Add output layer
ann.add(tf.keras.layers.Dense(units = 1, activation= 'sigmoid'))

# ps. hyper parameter : the parameter that won't be trained during the training process.

# Part III - Training the ANN

# Compile ANN with an optimizer, a loss function, and a metric with accuracy
ann.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training Set
# batch_size → in feature (training) dataset, we Sliced dataset in to some batches data
# so that we don't have to cal the loss function one-by-one data, use just adding all prediction of data in batch and compare real data in one time!)
# that's called batch learning
# epoch → how many times you train a whole dataset
print(X_train[0,:])
ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)



# Part IV - Make the predictions and evaluate the model

# Home Work : predict if one costomer will leave the bank
# Using the same step as Part I - data preprocessing    
# Note : If we only get one record data, we need to convert it into 2-D Array like we preprocess the 2-D DataFrame before
# Note II : We need to convert the data type after we Encoded the customer file we provided
# Note III : just remind the data type we get in every feature! some column's type was int, some column's type was float!
# Key : How we predict = How we train (initial data -> encoding -> scaling feature)

# IntroDuce data (2-D Array)
c_file = np.array([600,'France','Male',40,3,60000.0,2,1,1,50000.0], dtype='object')
# Use Label Encode and One-Hot Encode (Column Transformer) to deal with text feature
c_file[2] = le.transform([c_file[2]])[0]
C = ct.transform([c_file])
print(C)
# C = C.astype(float)
# Feature Scaling 
C = sc.transform(C) 
print(C)

# Predict the custom file
prediction = ann.predict(C)
print('Emotional DAAAAMAGE! The Custormer Will leave the Bank!' if prediction>0.5 else "That's how you do it! the Customer will NOT leave!")

# Predict the Test dataset
Y_pred = ann.predict(X_test)

# Convert the probability of Y_pred into binary value
Y_pred = (Y_pred >0.5)
print(Y_pred)
print(Y_test, type(Y_test))
# Compare prediction and real value to ensure if model predict correctly
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), np.array(Y_test).reshape(len(Y_test),1)),1))

# !!!! Create Confusion Matrix !!!!
# create the Confusion Matrix object
cm = confusion_matrix(np.array(Y_test), Y_pred)
print(cm)
score = accuracy_score(np.array(Y_test), Y_pred)
print(score)