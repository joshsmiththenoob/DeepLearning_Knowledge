# predict which person has highest risk to leave the bank from correlations between these features
# to prevent the "max-guests" from leaving back.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


print(tf.__version__)           # Test if tensorflow has installed

# Part I - Data Preprocessing

# Import dataset
Data_Table = pd.read_csv(r'./Churn_Modelling.csv')
X = Data_Table.iloc[:, 3:-1].values             # got table of features -> 2D NumpyArray -> decide to get which columns are features relevant to label
Y = Data_Table.iloc[:, -1]                      # got coressponding label -> 1D Series -> dependent variable on indepedent variable x (features)

# Encode categorical data
# Label Encode the "Gender column" ; Gender column -> X[:,2]
print(X[:, 2])
le = LabelEncoder()
# Label Encode - fit_transform : turn the text feature into a digi number feature like "Female -> 0, Male -> 1"
X[:, 2] = le.fit_transform(X[:, 2])

print(X[:,2])

# One-Hot Label Encode the "Geography" column -> X[:,1]
# Label Encode vs One-Hot Label Encode : One-Hot Label Encode does NOT have order and scale relationshiop between France,Spain and Germany
# use [1] in ColumnTransformerto One-Hot Label Encode -> X[:,1]
print(X,X.shape)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough' )
X = np.array(ct.fit_transform(X))
# the transformed column(s) become the leftmost columns to get new table
print(X,X.shape)


# Split the dataset into the Training set and Test set
# train_data : test_data = 0.8: 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)

# Feature Scaling (Compulsory)
S
