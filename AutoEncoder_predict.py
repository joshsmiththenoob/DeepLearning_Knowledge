# AutoEncoder 預估程式
# 使用支持向量機進行同樣的回歸預測 -> 以 SVM - Regression (SVR) 為基礎來評估、比較AutoEncoder的好壞 -> 如果SVM好，這個AutoEncoder就沒有價值了

'''
 建立SVR, 以做比較基準(baseline)
'''
# baseline in performance with support vector regression model
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

# 創建dataset
X_data, Y_data = make_regression(n_samples= 1000, n_features= 100, n_informative= 10, noise= 0.1, random_state= 1)
# 切割成訓練、測試集
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=1)
print(X_train.shape, Y_train.shape)
# reshape target variables so that we can transform them : MinMaxScaler吃的是二維矩陣(100,) -> (100,1)
Y_train = Y_train.reshape((len(Y_train), 1))
Y_test = Y_test.reshape((len(Y_test), 1))

# Normalize input data
trans_in = MinMaxScaler()
X_train = trans_in.fit_transform(X_train)
X_test = trans_in.transform(X_test)

# Normalize output data
trans_out = MinMaxScaler()
Y_train = trans_out.fit_transform(Y_train)
Y_test = trans_out.transform(Y_test)


# define model
model = SVR()

# SVR 訓練 : fit model on the training dataset
model.fit(X_train, Y_train)

# SVR 預測 : 從測試集訓練
yhat = model.predict(X_test)

# 預測結果去歸一化 : invert transforms so we can calculate errors
yhat = yhat.reshape((len(yhat), 1))
yhat = trans_out.inverse_transform(yhat)
Y_test = trans_out.inverse_transform(Y_test)
# 計算SVR預測出的誤差誤差 : calculate error (MAE)
score = mean_absolute_error(Y_test, yhat)
print('SVR在原始數據訓練後且預估的誤差 : ', score)



'''
使用訓練好的Encode Model做回歸預測
'''
# load the model from file
encoder = load_model('encoder.h5')

# 利用encoder 去轉換將raw data (100個特徵) 轉換至 瓶頸bottleneck的輸出向量(100個特徵向量)
# encode the train data
X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)

# 用這些轉換過後的特徵向量data去訓練SVR模型
# define model
model = SVR()
# 利用特徵向量其中的訓練集去訓練SVR model : fit model on the training dataset
model.fit(X_train_encode, Y_train)
# SVR訓練完後，再預測測試集(特徵向量)的結果 yhat
yhat = model.predict(X_test_encode)

Y_test = trans_out.transform(Y_test)
# 預測結果去歸一化 : invert transforms so we can calculate errors
yhat = yhat.reshape((len(yhat), 1))
yhat = trans_out.inverse_transform(yhat)
y_test = trans_out.inverse_transform(Y_test)
# 計算SVR預測出的誤差誤差 : calculate error (MAE)
score = mean_absolute_error(y_test, yhat)
print('多了Encoder進行特徵擷取後,所得出的SVR預估誤差 : ',score)