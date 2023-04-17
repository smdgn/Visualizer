from __future__ import annotations

import math
import numpy as np
from typing import Union, Optional

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Visualizer:
    # vis class that brings together
    # the base model (independent)
    # what to visualize (features, channels, nodes) ->
    # these are functions and can be defined without model knowledge
    # image params
    # optimization stuff (pentalties, functions
    def __init__(self,
                 model: keras.Model,
                 names: Union[list[str], str],
                 processing: Optional[str] = None):
        if isinstance(names, str):
            names = [names]
        self._num_layers_to_compute = len(names)
        self.processing = processing
        # define the extraction model
        layers = [model.get_layer(name).output for name in names]
        self.extraction_model = tf.keras.Model(inputs=model.input, outputs=layers)
        self.extraction_model.trainable = False
        # TODO: add processing
        # TODO: option to avoid border pixels
        # TODO: include noise reduction function
        self._EPSILON = 1e-8
        self._optimizer_config = None
        self.optimizer = None
        self.image = None

    def _remove_borders(self, activation):
        # future cases
        return activation[:, 2:-2, 2:-2, :]

    def compiled_loss(self, layer_activations):
        # future cases
        return None

    def generate_image(self, shape: tuple):
        noise = tf.random.uniform(shape=(1, *shape, 3), dtype=tf.float32)
        # TODO: move preprocessing of image to the model input pipeline
        self.image = tf.Variable(noise, trainable=True, name='image')


    def compile(self, optimizer):
        self.optimizer = optimizer
        self._optimizer_config = optimizer.get_config()

    def compute_loss(self):
        layer_activations = self.extraction_model(self.image, training=False)
        # pack activations into batches for efficient loss computation
        layer_activations = tf.stack(layer_activations, axis=0)
        loss = self.compiled_loss(layer_activations)
        return loss

    def _compute_single_filter_loss(self, input_image, filter_index):
        activations = self.extraction_model(input_image, training=False)
        activations = activations[..., filter_index]
        return tf.reduce_mean(activations)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[], dtype=tf.int32),)
        )
    def gradient_ascent_step(self,
                             filter_index: int):
        with tf.GradientTape() as tape:
            tape.watch(self.image)
            loss = self._compute_single_filter_loss(self.image.value(), filter_index)
        # Compute gradients.
        gradients = tape.gradient(loss, self.image)
        # Normalize gradients for stable ascent
        #gradients /= tf.math.reduce_std(gradients) + self._EPSILON
        gradients = tf.math.l2_normalize(gradients)
        #self.optimizer.apply_gradients(zip([gradients], [self.image]))
        self.image.assign_add(gradients*10)
        #img.assign(tf.clip_by_value(img, -1, 1))
        return loss

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.int32))
    )
    def gradient_ascent(self, filter_index, steps):
        loss = tf.constant(0.0)
        # gradient ascent operation for one image
        for s in tf.range(steps):
            loss = self.gradient_ascent_step(filter_index)
            print(f"Step {s}:{loss}")
        return loss

    def visualize_filter(self,
                         filter_index: int,
                         steps: int,
                         epochs: int,
                         image_shape: tuple[int, int],
                         upscaling_factor: Optional[float] = 1.2,
                         blur_per_epoch: bool = True):
        # generate noise
        backup_image = self.generate_image(shape=image_shape)
        # allocate a new Variable, use image as backing Tensor
        self.image = tf.Variable(backup_image, name="image", trainable=True)
        # iterate through epochs
        for epoch in range(epochs):
            print("traced once")
            loss = self.gradient_ascent(filter_index, steps)

            # reset optimizer state
            #self.optimizer = self.optimizer.from_config(self._optimizer_config)
            #.optimizer = keras.optimizers.Adam(learning_rate=0.1)
            print("first round")
            # update backup image
            backup_image = self.image.value()
            img_mean = tf.reduce_mean(backup_image)
            img_std = tf.math.reduce_std(backup_image)
            backup_image -= img_std
            backup_image /= img_std + 1e-5
            backup_image *= 0.15
            backup_image += 0.5
            backup_image = tf.clip_by_value(backup_image, clip_value_min=0, clip_value_max=1)*255.0
            if upscaling_factor is not None:
                # compute the new shape
                image_shape = tuple(math.ceil(s*upscaling_factor) for s in image_shape)
                backup_image = tf.image.resize(backup_image, image_shape,
                                               method=tf.image.ResizeMethod.BICUBIC)
            if blur_per_epoch:
                backup_image = tfa.image.gaussian_filter2d(backup_image, [5,5])
            # reallocate Variable
            self.image = tf.Variable(backup_image, name="image", trainable=True)

        return self.image.value()

    def deprocess_image(self, img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img

#import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import matplotlib.pyplot as plt
#from tensorflow import keras
#import visualizer
#model = keras.applications.InceptionV3(include_top=False, input_shape=(None, None, 3))
model_input = tf.keras.layers.Input(shape=(None, None, 3))
pre_process = tf.cast(model_input, tf.float32)
pre_process = keras.applications.inception_v3.preprocess_input(pre_process)
model = keras.applications.InceptionV3(input_tensor=pre_process,  include_top=False)
#model.summary()
#name = "block14_sepconv1"
name = "conv2d_60"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
vis = Visualizer(model, name)
vis.compile(optimizer=optimizer)
img = vis.visualize_filter(150, 20, 12, (56,56), 1.2, True)
new_image = vis.deprocess_image(img[0,...].numpy())
plt.imshow(new_image)
plt.show()

class OptimObject:
    def __init__(self,
                 shape,
                 name: Optional[str]):
        self.shape = shape
        # TODO: unique identifier
        self.var = self.image = tf.Variable(trainable=True, name='test')


