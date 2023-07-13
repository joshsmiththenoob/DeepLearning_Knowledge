import tensorflow as tf
import numpy as np
y_true = np.array([[0.,1.],[0.,0.]])
y_pred = np.array([[1.,1.],[1.,0.]])

MAE = tf.reduce_mean(tf.abs(y_true - y_pred))

print(MAE)

print(tf.abs(y_true - y_pred))