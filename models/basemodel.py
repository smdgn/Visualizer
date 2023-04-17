from __future__ import annotations
from typing import Callable, Tuple, Optional, Union

import tensorflow as tf
from tensorflow import keras


class Basemodel:

    """Basemodel defining the build path"""

    def __init__(self,
                 model: Callable,
                 preprocessing: Callable):
        """

        Args:
            model: callable keras or tensorflow model
            preprocessing: callable preprocessing layer that will be inserted before the model input
        """
        self.model = model
        self.preprocessing = preprocessing
        self._is_build = False

    def build(self,
              input_shape: Tuple[int, int, int] = (224, 224, 3),
              weights: Optional[str] = 'imagenet',
              trainable: bool = False) -> keras.Model:
        """ """
        # define the io pipeline
        model_input = tf.keras.layers.Input(shape=input_shape)
        pre_process = tf.cast(model_input, tf.float32)
        pre_process = self.preprocessing(pre_process)
        model = self.model(input_tensor=pre_process, weights=weights, include_top=True)
        if trainable:
            model.trainable = True
        else:
            model.trainable = False
            assert model.trainable_variables == []
        return model

    @staticmethod
    def create_extractor(model: keras.Model,
                         names: Union[list[str], str]):
        #TODO: maybe the extractor should dispatch different types of models e.g.
        # features, classes, single neurons


        if isinstance(names, str):
            names = [names]
        # define the extraction model
        layers = [model.get_layer(name).output for name in names]
        model = keras.models.Model(inputs=model.input, outputs=layers)
        return model



