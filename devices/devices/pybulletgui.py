"""

"""

import pybullet as p

from devices.device import InputDevice


class PyBulletGUI(InputDevice):
    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)

    @property
    def codename(self):
        return 'pbg'

    def _connect(self):
        assert p.isConnected(), 'PyBullet not connected.'
        self._axis_ids = []
        keys = list(self._action.keys())
        for i in range(6):
            self._axis_ids.append(
                p.addUserDebugParameter(
                    paramName=keys[i],
                    rangeMin=-1,
                    rangeMax=1,
                    startValue=0))
        self._is_connected = True

    def _disconnect(self):
        pass

    def _update(self):
        action = []
        for i in range(6):
            value = p.readUserDebugParameter(self._axis_ids[i])
            value *= self.pos_scaling if i < 3 else self.orn_scaling
            action.append(value)
        self.pose = action


if __name__ == '__main__':

    def printout(s):
        print(s.pose)


    s = PyBulletGUI()
    s.on_update = printout
    import time

    while True:
        time.sleep(1)
