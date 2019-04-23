#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-15
#SBATCH --time=4:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=catch_dqn
#SBATCH --output=./outputs/catch_dqn-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
RUN1=$(($SLURM_ARRAY_TASK_ID*2 - 1))
RUN2=$(($SLURM_ARRAY_TASK_ID*2))

python3 ./DQN_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env catcher -run_number $RUN1
python3 ./DQN_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env catcher -run_number $RUN2
deactivate
