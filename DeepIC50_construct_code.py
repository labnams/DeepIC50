#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from pandas import DataFrame
from datetime import datetime
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, AveragePooling1D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K


# In[ ]:


dataset = np.load('path/input_file.npz')


# In[ ]:


train_X, train_y = dataset['x'], dataset['y']


# In[ ]:


test_X, test_y = dataset['x_t'], dataset['y_t']


# In[ ]:


num_classes = 3
learning_rate = 0.0001
training_epochs = 100
batch_size = 25


# In[ ]:


train_X = np.expand_dims(train_X, axis=2)
test_X = np.expand_dims(test_X, axis=2)
print('train_X shape:', train_X.shape)
print(train_X.shape[0], 'train samples')
print(test_X.shape[0], 'test samples')


# In[ ]:


model = Sequential()

## for yoon # Input size should be [batch, 1d, 2d, ch] = (None, 1, 15000, 1)
model.add(Conv1D (kernel_size = 11, filters = 16, input_shape=(train_X.shape[1],train_X.shape[2]), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D (kernel_size = 11, filters = 16, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size = 2, strides = 2, padding='same'))
model.add(Conv1D (kernel_size = 11, filters = 32, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D (kernel_size = 11, filters = 32, padding='same'))
model.add(MaxPooling1D(pool_size = 2, strides = 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D (kernel_size = 11, filters = 64, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size = 2, strides = 2, padding='same'))
model.add(Conv1D (kernel_size = 11, filters = 64, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size = 2, strides = 2, padding='same'))

model.add(Flatten())

model.add(Dense (1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense (2048))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense (4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense (2048))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense (1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation = 'softmax',activity_regularizer=keras.regularizers.l2()  ))
model.compile( loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()


# In[ ]:


model_train = model.fit(train_X, train_y, batch_size=batch_size,epochs=training_epochs,verbose=1,
validation_data=(test_X, test_y))


# In[ ]:


# Option 2: Save/Load the Entire Model
from keras.models import load_model

# Creates a HDF5 file 'my_model.h5'
model.save('model_save_name.h5')

# Deletes the existing model
# del model  

# Returns a compiled model identical to the previous one
# model = load_model('191010_1080ti_ratio20_epoch100_add_batch.h5')


# In[ ]:


test_eval = model.evaluate(test_X, test_y, verbose=1)


# In[ ]:


test_eval


# In[ ]:


predicted_classes = model.predict(test_X)


# In[ ]:


pred_prob = model.predict_proba(test_X)


# In[ ]:


predicted_classes_arg = np.argmax(np.round(predicted_classes),axis=1)


# In[ ]:


predicted_classes_cate = keras.utils.to_categorical(predicted_classes_arg, num_classes)


# In[ ]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_y, predicted_classes_cate, target_names=target_names))


# In[ ]:


train_y_arg = np.argmax(train_y,axis=1)
test_y_arg = np.argmax(test_y,axis=1)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_y_arg, predicted_classes_arg, labels=[0,1,2])

