"""

"""

import pygame
from pygame.locals import *

from devices.device import InputDevice


class XBoxController(InputDevice):
    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)

    @property
    def codename(self):
        return 'xbc'

    def _connect(self):
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, \
            'No joystick found!'
        self._device = pygame.joystick.Joystick(0)
        self._device.init()
        self._z_directions = [1, 1]
        self._is_connected = True

    def _disconnect(self):
        pass

    def _update(self):
        action = []
        for event in pygame.event.get():
            if event.type == JOYBUTTONUP or event.type == JOYBUTTONDOWN:
                button_state = [self._device.get_button(i) for i in range(15)]
                self._z_directions[0] *= -1 if button_state[8] else 1
                self._z_directions[1] *= -1 if button_state[9] else 1
            if event.type == JOYAXISMOTION:
                action = [self._device.get_axis(i) for i in range(6)]
                x = action[0]
                y = action[1] * -1
                z = (action[4] + 1) / 2 * self._z_directions[0]
                rx = action[2]
                ry = action[3] * -1
                rz = (action[5] + 1) / 2 * self._z_directions[1]
                action = [x, y, z, rx, ry, rz]
        if len(action) == 0:
            return
        for i in range(6):
            action[i] = 0.0 if abs(action[i]) < 0.001 else action[i]  # hpf
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling
            action[i] = round(action[i], 5)
        self.pose = action


if __name__ == '__main__':

    def printout(s):
        print(s.pose)


    s = XBoxController()
    s.start()
    s.on_update = printout
    import time

    while True:
        time.sleep(1)
