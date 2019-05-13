#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=mc_dqn_small
#SBATCH --output=./outputs/mc_dqn_small-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

RUN1=$(($SLURM_ARRAY_TASK_ID*5 - 4))
RUN2=$(($SLURM_ARRAY_TASK_ID*5 - 3))
RUN3=$(($SLURM_ARRAY_TASK_ID*5 - 2))
RUN4=$(($SLURM_ARRAY_TASK_ID*5 - 1))
RUN5=$(($SLURM_ARRAY_TASK_ID*5))

python3 ./DQN_Experiment.py -env mountain_car -small_network -run_number $RUN1 \
-buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR
python3 ./DQN_Experiment.py -env mountain_car -small_network -run_number $RUN2 \
-buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR
python3 ./DQN_Experiment.py -env mountain_car -small_network -run_number $RUN3 \
-buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR
python3 ./DQN_Experiment.py -env mountain_car -small_network -run_number $RUN4 \
-buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR
python3 ./DQN_Experiment.py -env mountain_car -small_network -run_number $RUN5 \
-buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR
deactivate
