import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 生成線性可分的數據
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.25)

# 建立模型
model = Sequential([
    Dense(1, input_dim=2, activation='sigmoid')
])

# 編譯模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模
model.fit(X, y, epochs=100, verbose=1)

# 視覺化數據及分割線
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

# 繪製分割線
xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1)
weights = model.layers[0].get_weights()
yy = (-weights[0][0] * xx - weights[1]) / weights[0][1]
plt.plot(xx, yy, 'k-')

plt.title("Dense Layer Classifier with TensorFlow/Keras")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
