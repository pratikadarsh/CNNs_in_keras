import keras
import tensorflow as tf
import numpy as np

from keras import Model, utils
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D

# Load Data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Load and Compile Model.
base_model = VGG16(include_top=False, input_shape=(32,32,3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128)(x)
x = Dense(32)(x)
logits = Dense(10, activation='softmax')(x) #83511963
model = Model(inputs=base_model.input, outputs=logits)
#model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train.
history = model.fit(x=x_train, y=y_train, batch_size=8, validation_split=0.05)
model.save('weights/vgg16.hdf5')

