# AutoEncoder : Learn a Compressed representation of raw data.
# Encoder : compresses the input
# Decoder : recreate the compressed input provided by the encoder

# 訓練過後 : 留下 Encoder， Decoder會被丟棄 -> 非監督式學習
# X = model.predict(X)

# Autoencoder for resgression

# 使用 Autoencoder 訓練 1000筆資料，每筆資料有100個特徵

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
# define dataset
X_data,Y_data = make_regression(n_samples= 1000, n_features= 100, n_informative=10, noise= 0.1, random_state= 1)
# summarize the dataset
print(type(X_data), type(Y_data))


# Part I . Data Prerpcoessing

# 1. Split into train set and test set
X_train, X_test, Y_train, Y_test = train