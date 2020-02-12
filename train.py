#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

import ray
from ray.tests.cluster_utils import Cluster 
from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0

Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml

Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml

Note that -f overrides all other trial-specific command-line options.
"""


# additional libraries for dynamic experience replay 
from ray.tune.registry import register_env, register_trainable
from ray.rllib.agents.registry import get_agent_class
import os
import random
import envs_launcher
import utilities as util
import shutil
from collections import deque
import pickle

#========================================
# Callback functions
# for (1) custom metric -> success rate,
# (2) for save successful robot demos  
#========================================
def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["success"] = 0

def on_episode_step(info):    
    pass

def on_episode_end(info):
    episode = info["episode"]
    if len(episode.last_info_for().values()) > 0:
        episode.user_data["success"] = list(episode.last_info_for().values())[0]
        episode.custom_metrics["successful_rate"] = episode.user_data["success"]

def on_sample_end(info):    
    pass

def on_postprocess_traj(info):
    pass 
    # if list(info["episode"].last_info_for().values())[0] > 0:
    #    save_episode(info["post_batch"])
    
def on_train_result(info):
    if "successful_rate_mean" in info["result"]["custom_metrics"]:
        info["result"]["successful_rate"] = info["result"]["custom_metrics"]["successful_rate_mean"]

def save_episode(samples):
    memory = deque()
    for row in samples.rows():
        obs = row["obs"]
        action = row["actions"]
        reward = row["rewards"]
        new_obs = row["new_obs"]
        done = row["dones"]
        memory.append((obs, action, reward, new_obs, done))
    # save transitions
    file_name = dir_path + str(random.random())
    out_file = open(file_name, 'wb')
    pickle.dump(memory, out_file)
    out_file.close()
    util.prGreen("A successful transition is saved, length {}".format(len(memory)))

def get_task_path(yaml_file):
    with open(yaml_file) as f:
        experiments = yaml.safe_load(f)
        experiment_name = next(iter(experiments))
        dir_path = experiments[experiment_name]["local_dir"]
        dir_path = os.path.expanduser(dir_path)
        dir_path = os.path.join(dir_path, experiment_name) + "/robot_demos/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    return dir_path   
#======================================
# End of Callback functions 
#======================================


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="--redis-max-memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-memory",
        default=None,
        type=int,
        help="--memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Whether to attempt to enable TF eager execution.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser

def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)

            # add callbacks for self-defined metric
            # and save successful transitions from RL agents 
            experiment_name = next(iter(experiments))
            experiments[experiment_name]["config"]["optimizer"]["robot_demo_path"] = dir_path            
            experiments[experiment_name]["config"]["callbacks"] = {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                    "on_sample_end": on_sample_end,
                    "on_train_result": on_train_result,
                    "on_postprocess_traj": on_postprocess_traj
                    }
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    for exp in experiments.values():
        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")
        if args.eager:
            exp["config"]["eager"] = True
        if args.trace:
            if not exp["config"].get("eager"):
                raise ValueError("Must enable --eager to enable tracing.")
            exp["config"]["eager_tracing"] = True

    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory)
        ray.init(address=cluster.address) #, log_to_driver=False)
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus)
            # log_to_driver=False) # disable the loggings
                                 # https://github.com/ray-project/ray/issues/5048 
    
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    random.seed(12345)
    
    # register customized envs
    register_env("ROBOTIC_ASSEMBLY", envs_launcher.env_creator)

    # register customized algorithms
    # register_trainable("APEX_DDPG_DEMO", get_agent_class("contrib/APEX_DDPG_DEMO"))

    # get the path for saving robot demos 
    dir_path = get_task_path(args.config_file)
    
    run(args, parser)
