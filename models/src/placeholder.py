
import tensorflow as tf


class Scope:
    def __init__(self):
        pass


class Layer(Scope):
    def __init__(self):
        super(Layer, self).__init__()


class Channel(Layer):
    def __init__(self):
        super(Channel, self).__init__()

    def loss(self):
        activations = activations[..., filter_index]
        return tf.reduce_mean(activations)


class Neuron(Channel):
    def __init__(self):
        super(Neuron, self).__init__()
