from abc import ABC, abstractmethod

from devices.callbacks import handle, validate


class ConnectPattern(ABC):
    def __init__(self):
        self._on_connect = None
        if '_is_connected' not in self.__dict__:
            self._is_connected = False

    @property
    def is_connected(self):
        """
        Get connection state.
        :return: bool, True if connected else False.
        """
        return self._is_connected

    def connect(self, *args, **kwargs):
        """
        Connect method.
        :param args: arguments
        :param kwargs: keyword arguments
        :return: bool, result
        """
        result = self._connect(*args, **kwargs)
        self._is_connected = result
        handle(self._on_connect, self)
        return result

    @abstractmethod
    def _connect(self, *args, **kwargs):
        """
        Called by public method.
        :param args: arguments
        :param kwargs: keyword arguments
        :return: bool, result
        """
        pass

    @property
    def on_connect(self):
        """
        Get callback.
        :return: func, callback
        """
        return self._on_connect

    @on_connect.setter
    def on_connect(self, func):
        """
        Set callback.
        :param func: func, callback
        """
        validate(func, allow_args=True, allow_return=True)
        self._on_connect = func
