# Autodesk Deep Reinforcement Learning for Robotic Assembly

Here we provide a sim-to-real RL training and testing environment for robotic assembly, as well as a modification to APEX-DDPG by adding the option of recording and using human demonstrations. We include two examples in simulation (pybullet), Franka Panda robot performing the peg-in-hole task and a robot-less end-effector performing the lap joint task. We also include a template for connecting your real robot, as you roll out a successfully learned policy. 

## Installation

This repository is tested to be compatible with Ray 0.7.5. Hence, the following instruction is for working with Ray 0.7.5. Please feel free to try later versions of Ray and modify the code accordingly.<br> 

1. Install the conda environment for ray 0.7.5: https://pypi.org/project/ray/. Use Python 3.6.
2. Install the following dependencies:
    * [pybullet](https://pypi.org/project/pybullet/)
    * [tensorflow](https://pypi.org/project/tensorflow/)
    * [getch](https://pypi.org/project/getch/)
    * [pygame](https://pypi.org/project/pygame/)
    * [transforms3d](https://pypi.org/project/transforms3d/)
    
    ```
    $ pip install pybullet==2.2.6
    $ pip install tensorflow==1.10.0
    $ pip install getch
    $ pip install pygame
    $ pip install transforms3d
     ```
3. Download the ray source code from https://github.com/ray-project/ray/releases/tag/ray-0.7.5 and keep the rllib folder in your local working directory.
4. Copy this file https://github.com/ray-project/ray/blob/releases/0.7.5/python/ray/setup-dev.py to the directory where the rllib folder is in. Edit line 46 to point to the local rllib folder as needed.  Run `setup-dev.py` to link to the local rllib. 
5. Clone this repository inside the rllib folder.
6. Run `copy-to-rllib.py` to install the patch. 

## Run

Configure the parameters:
- Environment parameters in `envs_launcher.py`
- Training hyper-parameters in `hyper_parameters/*.yaml` file.

Test and visualize the simulation environment with default input device:
```
python run.py 
```

Provide a demonstration with the xbox controller:

```
python run.py --input-type xbc --save-demo-data=True --demo-data-path=human_demo_data/<example>
```

Train a model:

```
python train.py -f hyper_parameters/*.yaml
```

Roll out a model:

```
python rollout.py <path_to_trained_model>/checkpoint-<iteration>/checkpoint-<iteration>
```

## Notes

- All code is written in [Python 3.6](https://www.python.org/).
- All code was tested on MacOS, Windows, and Ubuntu.
- For licensing information see [LICENSE](LICENSE.md).