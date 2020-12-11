import tensorflow as tf
import keras
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D

import models

class CnnModel():
    def __init__(self, model='vgg16', num_classes=10):
        pass

    def get_model(self, model_name, num_classes=10):
        model_func = getattr(self, model_name, None)
        if model_func is None:
            print("Warning !!! Model could not be retrieved.")
            return None
        else:
            return model_func(num_classes)

    def vgg16(self, num_classes=10):
        base_model = VGG16(include_top=False, input_shape=(32,32,3))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128)(x)
        x = Dense(32)(x)
        logits = Dense(10, activation='softmax')(x) #83511963
        model = Model(inputs=base_model.input, outputs=logits)
        return model

    def alexnet(self, num_classes=10):
        return models.alexnet.AlexNet(num_classes)
