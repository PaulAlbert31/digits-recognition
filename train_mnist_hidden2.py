#!/usr/bin/env python3

import os
#Data formatting
from keras.datasets import mnist
from keras.utils import to_categorical
#Linear classifier
from keras.layers import Input, Dense, Activation, Lambda, Dropout
from keras.models import Model, load_model
#Visu
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
dropout = True
summary = True
normalization = True

#Visualisation callbacks                
run_name = "hidden2"
logpath = generate_unique_logpath("./logs_hidden2", run_name)
checkpoint_filepath = os.path.join(logpath,  "best_model.h5")
tbcb = TensorBoard(log_dir=logpath)
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)

#loading Mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_train = X_train.shape[0]
num_test = X_test.shape[0]

img_height = X_train.shape[1]
img_width = X_train.shape[2]

#Reshaping to 1-dim
X_train = X_train.reshape((num_train,img_width*img_height))
X_test = X_test.reshape((num_test,img_width*img_height))

#Converting to one-hot encoding
y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)

#Layers creation
num_classes = 10
xi = Input(shape=(img_height*img_width,))
xl = xi

if (normalization):
        #Optional normalization of the dataset
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)+1e-5
        xl = Lambda(lambda image, mu, std:(image - mu) / std,
                    arguments = {'mu':mean, 'std':std})(xi)

#Hidden layers and using dropout on input (20%) and on hidden layers(50%) to prevent overfiting
#nb of neurons on hidden layers
nhidden1 = 256
nhidden2 = nhidden1
if (dropout):
        xl = Dropout(0.2)(xl)
x = Dense(nhidden1)(xl)
x = Activation('relu')(x)
if (dropout):
        x = Dropout(0.5)(x)
x = Dense(nhidden2)(x)
x = Activation('relu')(x)
if (dropout):
        x = Dropout(0.5)(x)
x0 = Dense(num_classes)(x)
yo = Activation('softmax')(x0)
model = Model(inputs=[xi],outputs=[yo])

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
