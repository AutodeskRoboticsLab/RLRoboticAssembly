"""
Test simulated envs or Collect human demonstration data
"""

import argparse
import collections
import pickle
import numpy as np

import envs_launcher
import devices
import utilities


def getargs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-type',
        type=str,
        default='pbg',
        help='Input device type; must be equal to any of: ' +
             ', '.join(devices.REGISTRY.keys()))
    parser.add_argument(
        '--input-scaling',
        type=tuple,
        default=(10, 25),
        help='Scaling applied to device inputs; (pos, orn).')
    parser.add_argument(
        '--action-space',
        type=int,
        default=6,
        help='Degrees of freedom; must be equal to either 3 or 6.')
    parser.add_argument(
        '--save-demo-data',
        type=bool,
        default=False,
        help='Save demonstration data.')
    parser.add_argument(
        '--demo-data-path',
        type=str,
        default='human_demo_data/default',
        help='Location to save demonstration data')
    args = parser.parse_args()
    assert args.input_type in devices.REGISTRY, \
        'arg `input-type` must be equal to any of: ' + ', '.join(devices.REGISTRY.keys())
    assert args.action_space == 3 or args.action_space == 6, \
        'arg `action-space` must be equal to either 3 or 6.'
    return args


def main():
    args = getargs()

    environment = envs_launcher.env_creator(args)
    obs = environment.reset()
    memory = collections.deque()

    device_cls = devices.REGISTRY[args.input_type]
    device = device_cls(*args.input_scaling)
    device.start()

    done = False
    while not done:
        device.update()
        action = device.pose[:args.action_space]

        new_obs, reward, done, info = environment.step(action)

        if args.save_demo_data and np.count_nonzero(action) > 0:  # we don't want to save all-zero actions
            memory.append((obs, action, reward, new_obs, done))
            obs = new_obs
    else:
        device.disconnect()

    if args.save_demo_data:
        # save all the transitions
        file_name = args.demo_data_path
        out_file = open(file_name, 'wb')
        pickle.dump(memory, out_file)
        out_file.close()
        utilities.prGreen('Transition saved')
        utilities.prGreen('Steps: {}'.format(len(memory)))


if __name__ == '__main__':
    main()
