from tensorflow import keras

from models.basemodel import Basemodel


def resnet50():
    return Basemodel(keras.applications.resnet50.ResNet50,
                     keras.applications.resnet50.preprocess_input
                     )


def resnet50v2():
    return Basemodel(keras.applications.resnet_v2.ResNet50V2,
                     keras.applications.resnet_v2.preprocess_input
                     )


def resnet101v2():
    return Basemodel(keras.applications.resnet_v2.ResNet101V2,
                     keras.applications.resnet_v2.preprocess_input
                     )
