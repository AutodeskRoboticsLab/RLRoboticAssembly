"""

"""
import warnings

from devices.devices.nothing import Nothing
from devices.devices.pybulletgui import PyBulletGUI
from devices.devices.spacemouse import SpaceMouse
from devices.devices.xboxcontroller import XBoxController

# include classes in registry
REGISTRY = {}
for cls in [Nothing, PyBulletGUI, SpaceMouse, XBoxController]:
    # noinspection PyBroadException
    try:
        d = cls()
        REGISTRY[d.codename] = cls
        REGISTRY[d.clsname] = cls
    except Exception as e:
        warnings.warn('{} could not be registered due to '
                      'Exception: {}'.format(cls.__name__, e))
