from abc import ABC, abstractmethod

from devices.callbacks import handle, validate


class DisconnectPattern(ABC):
    def __init__(self):
        self._on_disconnect = None
        if '_is_connected' not in self.__dict__:
            self._is_connected = False

    def disconnect(self, *args, **kwargs):
        """
        Disconnect method.
        :param args: arguments
        :param kwargs: keyword arguments
        :return: bool, result
        """
        result = self._disconnect(*args, **kwargs)
        self._is_connected = result
        handle(self._on_disconnect, self)
        return result

    @abstractmethod
    def _disconnect(self, *args, **kwargs):
        """
        Called by public method.
        :param args: arguments
        :param kwargs: keyword arguments
        :return: bool, result
        """
        pass

    @property
    def on_disconnect(self):
        """
        Get callback.
        :return: func, callback
        """
        return self._on_disconnect

    @on_disconnect.setter
    def on_disconnect(self, func):
        """
        Set callback.
        :param func: func, callback
        """
        validate(func, allow_args=True, allow_return=True)
        self._on_disconnect = func