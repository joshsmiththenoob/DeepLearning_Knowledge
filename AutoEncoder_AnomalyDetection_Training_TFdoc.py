# Recall : autoencoder is trained to Minimize Reconstruction Error
# You will train an autoencoder on the normal rhythms only,
# Then use it to reconstruct all the data

# 載入心電圖 : Load ECG data (Electrocardiograms)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
print(dataframe.head())

# 心電圖csv最後一欄是label : 0 = 不正常 abnormal rhythm, 1 = 正常 normal rhythm
labels = raw_data[:, -1]

# 其他欄位是心電特徵
data = raw_data[:, 0:-1]

# Part I . Data PreProcessing
# 分割數據成訓練集，測試集
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size = 0.2, random_state = 21
)





