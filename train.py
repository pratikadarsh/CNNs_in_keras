import keras
import tensorflow as tf
import numpy as np

from keras import Model
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense

# Load Data.
(xtrain, y_train), (x_test, y_test) = cifar10.load_data()

# Load and Compile Model.
base_model = VGG16(include_top=False)
x = Flatten()(base_model.input)
x = Dense(128)(x)
x = Dense(32)(x)
logits = Dense(10, activation='softmax')(x) #83511963
model = Model(inputs=base_model.input, outputs=logits)
model.summary()
# Train.

