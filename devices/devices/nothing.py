"""

"""

import random

from devices.device import InputDevice


class Nothing(InputDevice):
    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)
        self.is_random = True

    @property
    def codename(self):
        return 'nth'

    def _connect(self):
        self._is_connected = True

    def _disconnect(self):
        self._is_connected = False

    def _update(self):
        action = []
        for i in range(6):
            a = random.uniform(-1.0, 1.0) if self.is_random else 0.0
            a *= self.pos_scaling if i < 3 else self.orn_scaling
            action.append(a)
        self.pose = action
