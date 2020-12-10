import keras
import tensorflow as tf
import numpy as np

from keras import utils
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D

from model import CnnModel

# Load Data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Load and Compile Model.
model = CnnModel().get_model('vgg16')
#model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train.
history = model.fit(x=x_train, y=y_train, batch_size=8, validation_split=0.05)
model.save('weights/vgg16.hdf5')

