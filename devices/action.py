#! usr/bin/env python
"""

"""

from threading import Event

X = 'x'
Y = 'y'
Z = 'z'
RX = 'rx'
RY = 'ry'
RZ = 'rz'


class Action:
    def __init__(self):
        """
        Base action class.
        """
        self._action = {}
        for name in [X, Y, Z, RX, RY, RZ]:
            self._action[name] = 0.0
        self._event = Event()

    def await_new(self, timeout=None):
        """
        Wait for this device to have new data.
        :param timeout: float or None, timeout
        :return: bool, True if data acquired before timeout else False
        """
        return self._event.wait(timeout)

    @property
    def has_new(self):
        """
        Get whether this device has new data.
        :return: bool, True if new data available else False
        """
        return self._event.isSet()

    @property
    def x(self):
        """
        Action attribute getter.
        :return: float
        """
        self._event.clear()
        return self._action[X]

    @x.setter
    def x(self, value):
        """
        Action attribute setter.
        :param value: float
        """
        self._event.set()
        self._action[X] = float(value)

    @property
    def y(self):
        """
        Action attribute getter.
        :return: float
        """
        self._event.clear()
        return self._action[Y]

    @y.setter
    def y(self, value):
        """
        Action attribute setter.
        :param value: float
        """
        self._event.set()
        self._action[Y] = float(value)

    @property
    def z(self):
        """
        Action attribute getter.
        :return: float
        """
        self._event.clear()
        return self._action[Z]

    @z.setter
    def z(self, value):
        """
        Action attribute setter.
        :param value: float
        """
        self._event.set()
        self._action[Z] = float(value)

    @property
    def rx(self):
        """
        Action attribute getter.
        :return: float
        """
        self._event.clear()
        return self._action[RX]

    @rx.setter
    def rx(self, value):
        """
        Action attribute setter.
        :param value: float
        """
        self._event.set()
        self._action[RX] = float(value)

    @property
    def ry(self):
        """
        Action attribute getter.
        :return: float
        """
        self._event.clear()
        return self._action[RY]

    @ry.setter
    def ry(self, value):
        """
        Action attribute setter.
        :param value: float
        """
        self._event.set()
        self._action[RY] = float(value)

    @property
    def rz(self):
        """
        Action attribute getter.
        :return: float
        """
        self._event.clear()
        return self._action[RZ]

    @rz.setter
    def rz(self, value):
        """
        Action attribute setter.
        :param value: float
        """
        self._event.set()
        self._action[RZ] = float(value)

    @property
    def pose(self):
        """
        Get pose.
        :return: tuple
        """
        self._event.clear()
        return [self._action[name] for name in [X, Y, Z, RX, RY, RZ]]

    @pose.setter
    def pose(self, tup):
        """
        Set pose.
        :param tup: tuple
        """
        for i, name in enumerate([X, Y, Z, RX, RY, RZ]):
            self._action[name] = float(tup[i])
        self._event.set()

    @property
    def pos(self):
        """
        Get position.
        :return: tuple
        """
        self._event.clear()
        return [self._action[name] for name in [X, Y, Z]]

    @pos.setter
    def pos(self, tup):
        """
        Set position.
        :param tup: tuple
        """
        for i, name in enumerate([X, Y, Z]):
            self._action[name] = float(tup[i])
        self._event.set()

    @property
    def orn(self):
        """
        Get orientation.
        :return: tuple
        """
        self._event.clear()
        return [self._action[name] for name in [RX, RY, RZ]]

    @orn.setter
    def orn(self, tup):
        """
        Set orientation.
        :param tup: tuple
        """
        for i, name in enumerate([RX, RY, RZ]):
            self._action[name] = float(tup[i])
        self._event.set()
