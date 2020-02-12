from abc import ABC, abstractmethod

from devices.callbacks import handle, validate


class UpdatePattern(ABC):
    def __init__(self):
        self._on_update = None

    def update(self, *args, **kwargs):
        """
        Update method.
        :param args: arguments
        :param kwargs: keyword arguments
        :return: bool, result
        """
        result = self._update(*args, **kwargs)
        handle(self._on_update, self)
        return result

    @abstractmethod
    def _update(self, *args, **kwargs):
        """
        Called by public method.
        :param args: arguments
        :param kwargs: keyword arguments
        :return: bool, result
        """
        pass

    @property
    def on_update(self):
        """
        Get callback.
        :return: func, callback
        """
        return self._on_update

    @on_update.setter
    def on_update(self, func):
        """
        Set callback.
        :param func: func, callback
        """
        validate(func, allow_args=True, allow_return=True)
        self._on_update = func
