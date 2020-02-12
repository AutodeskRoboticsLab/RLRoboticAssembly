# rllib patch 

from shutil import copyfile 
import os

# show all the stats in terminal, not just one third 
dqn_src = "copy_to_rllib/agents/dqn/dqn.py"
dqn_dst = "../agents/dqn/dqn.py"
copyfile(dqn_src, dqn_dst)

# human demonstrations & dynamic experience replay 
async_src = "copy_to_rllib/optimizers/async_replay_optimizer.py"
async_dst = "../optimizers/async_replay_optimizer.py"
copyfile(async_src, async_dst)
buffer_src = "copy_to_rllib/optimizers/replay_buffer.py"
buffer_dst = "../optimizers/replay_buffer.py"
copyfile(buffer_src, buffer_dst)

# calculate the custom metrics for one transition, 
# not the historic transitions designed by Ray  
metrics_src = "copy_to_rllib/evaluation/metrics.py"
metrics_dst = "../evaluation/metrics.py"
copyfile(metrics_src, metrics_dst) 



