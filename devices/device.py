#! usr/bin/env python
"""

"""

from abc import ABC

from devices.action import Action
from devices.pconnect import ConnectPattern
from devices.pdisconnect import DisconnectPattern
from devices.pupdate import UpdatePattern


class InputDevice(ConnectPattern, DisconnectPattern, UpdatePattern, Action, ABC):
    def __init__(self, pos_scaling=1.0, orn_scaling=1.0):
        """
        Initialize input device. The class methods Connect, Disconnect, and
        Update are abstract and must be implemented in all device subclasses.
        :param pos_scaling: float, scaling applied to input positions
        :param orn_scaling: float, scaling applied to input orientations
        """
        ConnectPattern.__init__(self)
        DisconnectPattern.__init__(self)
        UpdatePattern.__init__(self)
        Action.__init__(self)
        # Initialize properties
        self.action = Action()
        self.pos_scaling = pos_scaling
        self.orn_scaling = orn_scaling
        self._device = None

    @property
    def clsname(self):
        """
        Get the name of this device class.
        :return: str
        """
        return self.__class__.__name__.lower()

    @property
    def codename(self):
        """
        Get the codename of this device class.
        :return:
        """
        raise NotImplementedError

    def start(self):
        """
        Start this device.
        """
        if not self._is_connected:
            self.connect()
