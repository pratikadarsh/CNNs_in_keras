import tensorflow as tf
import keras
from keras.models import Sequential
from keras import Model
from keras.layers import Dense, Conv2D, MaxPooling2D


class AlexNet():
    def __init__(self, num_classes):
        model = self.alexnet(num_classes)

    def alexnet(num_classes):
        model = Sequential()
        model.add(Conv2D(96, 11, strides=(11,11), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(256, 5, strides=(1,1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(384, 3, strides=(1,1), padding='same', activation='relu'))
        model.add(Conv2D(384, 3, strides=(1,1), padding='same', activation='relu'))
        model.add(Conv2D(256, 3, strides=(1,1), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        return model
