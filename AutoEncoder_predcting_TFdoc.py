# AutoEncoder : 訓練重建原生data的能力 -> 平常的data我有信心回復平常data原來的樣子(reconstruction error最小化)；但異常的data我就沒辦法有信心回復了!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# 1. 分割資料成訓練、測試集
(X_train,_),(X_test, _) = fashion_mnist.load_data()

# 2. 歸一化(MinMax Normalization)
x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

# 定義AutoEncoder模型架構
n_bottle_neck = 64

class Autoencoder(Model): # 繼承Model的屬性
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__() # 繼承Model的方法
    self.latent_dim = latent_dim 
    # 建立 Encoder (Compressing the input)  
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    # 建立 Decoder (Reconstructing the compressed input Encoder made)
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='linear'),
      layers.Reshape((28, 28))
    ])

  def call(self, x): # pic -> encoding -> decoding
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  

def get_model():
  return Autoencoder(n_bottle_neck)

# 3. 新增模型架構並載入訓練好的權重
## 讀取方法I : 建立相同模型並載入權重
# checkpoint_path = 'Autencoder_training_1/ckpt'
# autoencoder = get_model()
# autoencoder.compile(optimizer='adam', loss='mse')
# # Initialize the variables used by optimizers
# autoencoder.train_on_batch(x_train[:1], x_train[:1])
# # Load the state of the Old model
# autoencoder.load_weights(checkpoint_path)

# 讀取方法II : 讀取整個模型，不用再建立一個新的模型架構
model_path = 'Autencoder_training_1/ckpt.h5'
autoencoder = load_model(model_path)
# 4. 使用AutoEncoder 的 Encoder 以及 Decoder
decoded_imgs = np.array(autoencoder.predict(x_test))

# 5. 畫圖顯示
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(X_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()