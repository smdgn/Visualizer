from __future__ import annotations

import math
import numpy as np
from typing import Union, Optional

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from src.placeholder import Layer
from models.resnet import resnet50

class OptimObject:
    def __init__(self,
                 shape,
                 name: Optional[str],
                 initializer = tf.random_normal_initializer()):
        self.shape = shape
        # TODO: unique identifier
        # TODO different init support
        # TODO: batch support
        # TODO: image params
        self.var = tf.Variable(initializer(shape=(1, *shape, 3)),
                               trainable=True, name='test', dtype=tf.float32)


class Visualizer:

    # TODO: add epoch
    # TODO: add epoch callbacks
    def __init__(self):
        self.optimizer = None
        self.compiled_loss = None
        self.gradient_penalty = None

    def visualize(self,
                  model,
                  scope,
                  optimobj,
                  steps=100):
        extractor = model.build((*optimobj.shape, 3))
        scope.build(extractor)
        self.gradient_ascent(scope, optimobj, steps=steps)
        return optimobj.var.value()


    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.int32),)
    )
    def gradient_ascent_step(self, scope, optimobj):
        with tf.GradientTape() as tape:
            tape.watch(optimobj.var)
            tensor = scope(optimobj.var)
            loss = self.compiled_loss(tensor)
        # Compute gradients.
        gradients = tape.gradient(loss, optimobj.var)
        # Normalize gradients for stable ascent
        gradients = self.gradient_penalty(gradients)
        self.optimizer.apply_gradients(zip([gradients], [optimobj.var]))
        return loss

    def gradient_ascent(self, scope, optimobj, steps):
        #loss = tf.constant(0.0)
        # gradient ascent operation for one image
        for s in tf.range(steps):
            loss = self.gradient_ascent_step(scope, optimobj)
            print(f"Step {s}:{loss}")

    def compile(self,
                optimizer,
                loss,
                gradient_penalty):
        self.optimizer = optimizer
        self.compiled_loss = loss
        self.gradient_penalty = gradient_penalty


def std_penalty(gradients):
    gradients /= tf.math.reduce_std(gradients) + 1e-6
    return gradients

def l2_penalty(gradients):
    gradients = tf.math.l2_normalize(gradients)
    return gradients

def mean_loss(tensor):
    return tf.reduce_mean(tensor)

def deprocess_image(img):
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

def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)


#import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import matplotlib.pyplot as plt

resnet = resnet50()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
name = "conv4_block4_1_conv"

scope = Layer(name)[20]
img = OptimObject((200, 200), name="test")
vis = Visualizer()

vis.compile(optimizer=optimizer, gradient_penalty=l2_penalty, loss=mean_loss)
new_img = vis.visualize(resnet, scope, img, steps=200)

plt.imshow(deprocess_image(new_img[0,:,:,:].numpy()))
plt.show()