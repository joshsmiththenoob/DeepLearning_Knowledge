# AutoEncoder : 訓練重建原生data的能力 -> 平常的data我有信心回復平常data原來的樣子(reconstruction error最小化)；但異常的data我就沒辦法有信心回復了!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from keras.models import Model
from tensorflow.keras.utils import plot_model



def get_model(latent_dim):
    # Encoder
    input = layers.Input(shape=(28,28), name = 'input')
    e = layers.Flatten()(input)
    e = layers.Dense(latent_dim, activation = 'relu',name = 'bottle_neck')(e)
    # Decoder
    d = layers.Dense(784, activation = 'linear')(e)
    output = layers.Reshape((28,28))(d)
    autoencoder_model = Model(inputs = [input], outputs = output)
    return autoencoder_model


# Load dataset (28x28的黑白流行圖片，不用Y標籤)
# Part I : Data PreProcessing
# 1. 分割資料成訓練、測試集
(X_train,_),(X_test, _) = fashion_mnist.load_data()

# 2. 歸一化(MinMax Normalization)
x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)



# 定義AutoEncoder模型
n_bottle_neck = 64


autoencoder = get_model(n_bottle_neck)
autoencoder.compile(optimizer='adam', loss='mse')
# Initialize the variables used by optimizers
# autoencoder.train_on_batch(x_train[:1],x_train[:1])
# 紀錄模型儲存資訊 (紀錄最後一個epoch時的權重)
checkpoint_path = 'Autencoder_training_1/ckpt.h5'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 verbose=1
                                                 )

# 4. 畫出模型架構圖
plot_model(autoencoder, 'autoencoder_tf.png', show_shapes = True)

# 使用X_train進行訓練，特徵 = X_train, 標籤 = X_train，讓AutoEncoder學習這些圖片的特徵，並完整還原
history = autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks = [cp_callback])


# plot loss : 查看學習狀況
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# 4. 使用AutoEncoder 的 Encoder 以及 Decoder
decoded_imgs = np.array(autoencoder.predict(X_test))


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