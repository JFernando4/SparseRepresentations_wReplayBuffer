#!/usr/bin/env bash

export PYTHONPATH=.
for i in 1 3 4 5 6 7 8 9
do
    echo "Run $i..."
    python3 ./DQN_Experiment.py -buffer_size 20000 -tnet_update_freq 10 -verbose -lr 0.01 -run_number $i
done

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# buffer size = {10k, 20k, 40k}
# target network update frequency = {10, 50, 100, 200, 400}