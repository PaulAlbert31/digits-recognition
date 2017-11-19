#!/usr/bin/env python3
import os
#Data formatting
from keras.datasets import mnist
from keras.utils import to_categorical
#Classifiers
from keras.layers import Input, Dense, Activation, Lambda, Dropout, Flatten
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

#Visualisation
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

#Unique path to store TensorBoard data
def generate_unique_logpath(logdir, raw_run_name):
        i = 0
        while(True):
                run_name = raw_run_name + "-" + str(i)
                log_path = os.path.join(logdir, run_name)
                if not os.path.isdir(log_path):
                        return log_path
                i = i + 1


#Options, could be optimised for preprocessing
normalization = True
dropout = True
summary = True

#Visualisation callbacks                
run_name = "conv"
logpath = generate_unique_logpath("./logs_conv", run_name)
checkpoint_filepath = os.path.join(logpath,  "best_model.h5")
tbcb = TensorBoard(log_dir=logpath)
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_train = X_train.shape[0]
num_test = X_test.shape[0]

img_height = X_train.shape[1]
img_width = X_train.shape[2]

X_train = X_train.reshape([-1,img_height,img_width,1])
X_test = X_test.reshape([-1,img_height,img_width,1])

#Converting to one-hot encoding
y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)

#Layers creation
num_classes = 10
input_shape = (img_height, img_width, 1)
xi = Input(shape=input_shape)
xl = xi

#Optional normalization of the dataset
if (normalization):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)+1e-5
        xl = Lambda(lambda image, mu, std:(image - mu) / std,
            arguments = {'mu':mean, 'std':std})(xl)

#Input layer + Lambda normalization+ Dropout 20% + 3 consecutive (Conv+Relu+MaxPooling) + 2consectutive (Dropout 50%+Dense+ Relu) + Dense + softmax
if (dropout):
        xl = Dropout(0.2)(xl)
x = Conv2D(filters=16,
           kernel_size=5, strides=1,
           padding='same')(xl)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Conv2D(filters=32,
           kernel_size=5, strides=1,
           padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Conv2D(filters=64,
           kernel_size=5, strides=1,
           padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)
if (dropout):
        x = Dropout(0.5)(x)
x = Dense(128)(x)
x = Activation('relu')(x)
if (dropout):
        x = Dropout(0.5)(x)
x0 = Dense(64)(x)
x = Activation('relu')(x)
x = Flatten()(x)
x0 = Dense(num_classes,activation = 'softmax')(x)
model = Model(inputs=[xi],outputs=[x0])

if (summary):
        model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1,
          callbacks=[tbcb,checkpoint_cb])

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
