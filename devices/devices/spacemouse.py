#! usr/bin/env python
"""

"""

import pygame
from pygame.constants import JOYAXISMOTION

from devices.device import InputDevice


class SpaceMouse(InputDevice):
    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)

    @property
    def codename(self):
        return 'spm'

    def _connect(self):
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, \
            'No joystick found!'
        self._device = pygame.joystick.Joystick(0)
        self._device.init()
        self._is_connected = True

    def _disconnect(self, *args, **kwargs):
        pass

    def _update(self):
        action = []
        for event in pygame.event.get():
            if event.type == JOYAXISMOTION:
                action = [self._device.get_axis(i) for i in range(6)]
        if len(action) == 0:
            return
        for i in range(6):
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling
        action[0] = -action[0]  # x direction is positive
        self.pose = tuple(action)


if __name__ == '__main__':

    def printout(s):
        print(s.pose)


    s = SpaceMouse()
    s.start()
    s.on_update = printout
    import time

    while True:
        time.sleep(1)
