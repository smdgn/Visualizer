import functools
from typing import Iterable, Callable, Union, Type, Generic, TypeVar, Mapping
from functools import wraps, update_wrapper

import keras.applications.xception

T = TypeVar('T')


class Factory(Generic[T]):
    def __init__(self):
        self._builders = {}

    def _register_builder(self, key: str, builder: T, overwrite: bool):
        if key in self._builders.keys() and not overwrite:
            raise ValueError(f"Can't overwrite existing key")
        self._builders[key] = builder

    def register(self, func: T, name: str, *, overwrite: bool = False):
        self._register_builder(name, func, overwrite)

    def create(self, key: str, **kwargs):
        builder = self.get(key)
        return builder(**kwargs)

    def registered_objects(self) -> Mapping[str, T]:
        return self._builders

    def get(self, key: str, *, silent: bool = False) -> T:
        obj = self._builders.get(key)
        if not obj and not silent:
            raise ValueError(f"{key} not registered")
        return obj

    @property
    def types(self):
        """Evaluate registered keys and return as constants"""
        keys = list(self._builders.keys())
        return keys



