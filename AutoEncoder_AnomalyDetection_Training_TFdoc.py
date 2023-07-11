# Recall : autoencoder is trained to Minimize Reconstruction Error
# You will train an autoencoder on the normal rhythms only,
# Then use it to reconstruct all the data

# 載入心電圖 : Load ECG data (Electrocardiograms)
import pandas as pd

dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
print(dataframe.head())