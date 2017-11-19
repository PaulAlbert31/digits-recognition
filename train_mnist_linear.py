#!/usr/bin/env python3

import os
#Data formatting
from keras.datasets import mnist
from keras.utils import to_categorical
#Linear classifier
from keras.layers import Input, Dense, Activation, Lambda, Dropout
from keras.models import Model, load_model
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



#Options (could be optimised for preprocessing)
œnormalization = True
summary = True



#Visualisation callbacks                
run_name = "linear"
logpath = generate_unique_logpath("./logs_linear", run_name)
#Using modelcheckpoint to retain the best model
checkpoint_filepath = os.path.join(logpath,  "best_model.h5")
#Tensorboard will help us visualise the performances
tbcb = TensorBoard(log_dir=logpath)
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)

#Loading data from MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_train = X_train.shape[0]
num_test = X_test.shape[0]

img_height = X_train.shape[1]
img_width = X_train.shape[2]

#Reshaping to 1-dim (non convolutional)
X_train = X_train.reshape((num_train,img_width*img_height))
X_test = X_test.reshape((num_test,img_width*img_height))

#Converting to one-hot encoding
y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)

#Layers creation
num_classes = 10
xi = Input(shape=(img_height*img_width,))

if (normalization):
        #Optional normalization of the dataset
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)+1e-5 #avoiding values too close to 0
        #Using lambda function to produce standard score
        xl = Lambda(lambda image, mu, std:(image - mu) / std,
            arguments = {'mu':mean, 'std':std})(xi)
        x0 = Dense(num_classes)(xl)
else:
        x0 = Dense(num_clases)(xi)

yo = Activation('softmax')(x0)
model = Model(inputs=[xi],outputs=[yo])

if(summary):
        model.summary()

#Cross-entropic loss for classification
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
