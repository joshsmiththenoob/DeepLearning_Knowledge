# Dog and Cat classification
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
print(tf.__version__)
# Part I - Data Preprocessing

# Preprocessing the training set
# to separate training(Known) and Testing(Unknown) datatset
'''We just transformate only training dataset we Known -> Augument'''
'''-> Image Augumentation only on training dataset -> prevent overfitting'''
# Also, Feature scaling as rescaling the grayscale in ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
        r'.\Section 40 - Convolutional Neural Networks (CNN)\dataset\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') 
        # target_size : resize the source img that we can compute faster
        # batch_size : How many img s we want to have in each batch.
        # class_mode : to category the data we get -> binary or categorical

# Preprocessing the testing set
# only rescaling feature (pixel value : grayscale) on test dataset
test_datagen = ImageDataGenerator(rescale=1./255)


test_set = test_datagen.flow_from_directory(
        r'.\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set',
        target_size=(64, 64),
        batch_size=32, 
        class_mode='binary')


# Part II - Building Machine Learning
cnn = tf.keras.models.Sequential()
# Convolution layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation='relu',padding='valid',input_shape = [64,64,3]))
# Polling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
# Second Convolution layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation='relu',padding='valid'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
# Falttening
cnn.add(tf.keras.layers.Flatten())
# Full Connection Layer
cnn.add(tf.keras.layers.Dense(units = 128, activation= 'relu'))
# Output Layer - Classification for Possibility
cnn.add(tf.keras.layers.Dense(units = 1, activation= 'sigmoid'))


# Training the CNN

# Compile the CNN
cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = train_set, validation_data= test_set, epochs = 25)

# Part IV - Making a single prediction
'''Remind ! model was trained by feature scaled(grayscale scaled) and resized (64x64)img
-> !!! Don't forget to do get the same type as train data before predict any img !!!'''
# Read img as PIL format
test_img = image.load_img(r'.\Section 40 - Convolutional Neural Networks (CNN)\dataset\single_prediction\cat_or_dog_2.jpg'
                          , target_size = (64,64))
# Turn PIL format to np.array
test_img = image.img_to_array(test_img)
test_img = test_img/255.0

# Because we Train Batch of images !! so that batch dimension would be (batch_num, img_h,img_w)
'''That we Train the batch of imgs which has 3-D dimensions
-> !! Don't forget to set the same dimensions as train data !!! '''
# bath_num always on the first dimension
test_img = np.expand_dims(test_img,axis = 0)

result = cnn.predict(test_img)
print(train_set.class_indices)
if result[0][0] > 0.5 :
    prediction = 'dog'
    print('This is a dog!')
else:
    prediction = 'cat'
    print('This is a cat!')