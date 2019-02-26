#!/usr/bin/env bash

export PYTHONPATH=.
for i in 3 5 6
do
    echo "Run $i..."
    python3 ./DQN_Experiment.py -buffer_size 10000 -tnet_update_freq 200 -verbose -lr 0.001 -run_number $i -env acrobot
done

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# buffer size = {10k, 20k, 40k}
# target network update frequency = {10, 50, 100, 200, 400}