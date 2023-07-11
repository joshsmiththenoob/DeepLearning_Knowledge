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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
# define dataset
X_data,Y_data = make_regression(n_samples= 1000, n_features= 100, n_informative=10, noise= 0.1, random_state= 1) # n_informative : 真正有作用的特徵  ; noise : 高斯雜訊(noise = 高斯分布的標準差 ->影響真實值的分布) ;
# summarize the dataset
print(X_data.shape, Y_data.shape)


# Part I . Data Prerpcoessing

# 1. Split into train set and test set : 默認方法為隨機分割 (數據及打亂，按照比例分割 = 隨機抽樣)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=1)
n_inputs = X_data.shape[1]

# 2. Standarization or Normalization 標準化(Z-Score) or 歸一化(Min-Max Normalization)
MNScaler = MinMaxScaler()
X_train = MNScaler.fit_transform(X_train) # 先學習train的歸一化特徵(Min,Max)，再對train進行歸一化
X_test = MNScaler.transform(X_test) # 因為已經學習train的歸一化特徵，所以用同樣(模型)的標準對test歸一化


# Part II . Define AutoEncoder
# 1. 定義 Encoder : Compressing the input into feature vector
visible = Input(shape=(n_inputs,)) # 設置 Input 層， 輸入形狀 = (n_inputs,) = Dense神經元數量
e = Dense(n_inputs*2)(visible) # 設置一個Dense層，神經元為n_input的兩倍，輸出是e(張量(矩陣np.array))
# 以上code相當於 :
# model = Sequential()
# model.add(Dense(units = n_inputs*2,  input_shape = (n_inputs,)))

e = BatchNormalization()(e) # 在Activation Function前使用BatChNormalization -> Activation(BN(wx+b))，有助於在多層之後Activation funciton不在局限於一定的Range之間
e = ReLU()(e)

# define bottleneck # 設置最壓縮層(產生特徵向量)
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)


# 2. 定義 Decoder : reconstruct input from the compressed input encoder made
# 一層hidden Layer ， 接bottleneck -> ReLU(BN(wx+b))
d = Dense(n_inputs*2)(bottleneck)
d = BatchNormalization()(d)
d = ReLU()(d)

# output layer
output = Dense(n_inputs, activation = 'linear')(d)

# 3. 定義完整的AutoEncoder = Encoder + Decoder
# define autoencoder model
model = Model(inputs=visible, outputs = output)
# compile autoencoder model
model.compile(optimizer = 'adam', loss = 'mse')

# 4. 畫出模型架構圖
plot_model(model, 'autoencoder.png', show_shapes = True)


# Part III : Training AutoEncoder Model
# fit the autoencoder model to reconstruct input
history  = model.fit(X_train, X_train, epochs = 400, batch_size = 16, verbose =2, validation_data = (X_test, X_test))

# plot loss : 查看學習狀況
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# Part IV : 儲存Encoder model (without the decoder)
# define an encoder model
encoder = Model(inputs=visible, outputs = bottleneck)
plot_model(encoder, 'encoder.png', show_shapes = True)
# save the encoder to file
encoder.save_weights('encoder.h5')
