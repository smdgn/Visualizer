from __future__ import annotations
import abc
from typing import Tuple, Type, TypeVar, Union
from collections import OrderedDict
from functools import partial


import tensorflow as tf
from tensorflow import keras

BaseScope = TypeVar('BaseScope', bound='Scope')
#BaseScopeT = Tuple[BaseScope, BaseScope]


class Scope(metaclass=abc.ABCMeta):
    def __init__(self,
                 name: Union[list[str], str]):
        if isinstance(name, str):
            name = [name]
        self.layers = OrderedDict.fromkeys(name, type(self))

    @property
    @abc.abstractmethod
    def hierarchy(self) -> BaseScope:
        raise NotImplementedError("hierarchy property not defined")

    @classmethod
    def __subclasshook__(cls, subclass):
        # TODO: define subclasshook
        return (hasattr(subclass, 'load_data_source') and
                callable(subclass.load_data_source) and
                hasattr(subclass, 'extract_text') and
                callable(subclass.extract_text))

    def build(self, model: keras.Model) -> keras.Model:
        layers = [model.get_layer(name).output for name in list(self.layers.keys())]
        model = keras.models.Model(inputs=model.input, outputs=layers)
        return model

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        # TODO: add tensor type via tensorflow extension types
        """Define the scope access"""
        raise NotImplementedError()

    def __call__(self, tensor, *args, **kwargs):
        """Process list of tensors into one Tensor"""
        # check possible tensor states
        # 1. Single Tensor [B, H, W, C] -> Nothing to do
        if len(tensor) == 1:
            pass
        else:
            # List of tensors [[B, H, W, C], [[B, H, W, C]]
            batch, height, width, channel = zip(*[list(*t.shape()) for t in tensor])
            # 3. Check for similar h,w dimensions and create uniform tensor
            if all(h == height[0] for h in height) and all(w == width[0] for w in width):
                # same dimensions -> check if concat by channel or batch
                # for now just concat by channel
                tensor = tf.concat([t.reshape((1,height[0], width[0], -1)) for t in tensor], axis=-1)
            # 4. create a ragged tensor
            tensor = tf.ragged.constant(tensor)
            tf.get_logger().info(f"Created ragged tensor of shape {tensor.shape.as_list()} during"
                                 f"scope call")
        return self.call(tensor, *args, **kwargs)

    def __add__(self,
                other: BaseScope):
        self.layers.update(other)
        return self

    def __sub__(self,
                other: BaseScope):
        for k in other.layers:
            self.layers.pop(k, None)
        return self

    def __truediv__(self, other):
        pass


class Block(Scope):

    def __init__(self,
                 name: Union[list[str], str]):
        super(Block, self).__init__(name=name)
        if len(self.layers) != 1 and len(self.layers) % 2 != 0:
            raise ValueError("names are not divisible into pairs of two")

    def _unpack(self, model: keras.Model, layer_list: list, end:str=None, begin:str=None, *args):
        if end is None:
            return layer_list
        end = model.layers.index(end)
        start = model.layers.index(begin) if begin is not None else 0
        layer_list.extend(model.layers[start, end])
        self._unpack(model, layer_list,  *args)

    def build(self, model: keras.Model) -> keras.Model:
        layer_list = []
        layer_list = self._unpack(model, layer_list, *list(self.layers.keys()))
        layers = [model.get_layer(name).output for name in layer_list]
        model = keras.models.Model(inputs=model.input, outputs=layers)
        return model

    @property
    def hierarchy(self) -> None:
        return None

    def call(self, *args, **kwargs):
        # No special access operation
        pass


class Layer(Scope):

    def __init__(self,
                 name: Union[list[str], str]):
        super(Layer, self).__init__(name=name)

    def call(self, tensor, *args, **kwargs):
        # No special access operation
        pass

    def __getitem__(self,
                    item: int) -> Channel:
        temp_ch = Channel(list(self.layers.keys()))
        temp_ch.call = partial(temp_ch.call, channel_index=item)
        return temp_ch

    @property
    def hierarchy(self) -> Type[Block]:
        return Block


class Channel(Scope):

    def __init__(self,
                 name: Union[list[str], str]):
        super(Channel, self).__init__(name=name)

    def call(self, tensor, channel_index, *args, **kwargs):
        return tensor[..., channel_index]

    @property
    def hierarchy(self) -> Tuple[Type[Layer], Type[Neuron]]:
        return Layer, Neuron

    def __getitem__(self,
                    item: tuple[int, int]) -> Neuron:
        temp_n = Neuron(list(self.layers.keys()))
        temp_n.call = partial(temp_n.call, position=item)
        return temp_n


class Neuron(Scope):

    def __init__(self,
                 name: Union[list[str], str]):
        super(Neuron, self).__init__(name=name)

    def call(self, tensor, position, *args, **kwargs):
        height, width = position
        return tensor[:, height, width, :]

    @property
    def hierarchy(self) -> None:
        return None
