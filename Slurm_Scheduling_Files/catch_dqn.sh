#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-10
#SBATCH --time=4:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=catch_dqn
#SBATCH --output=./outputs/catch_dqn-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
first_run=$(($SLURM_ARRAY_TASK_ID*3 - 2))
second_run=$(($SLURM_ARRAY_TASK_ID*3 - 1))
third_run=$(($SLURM_ARRAY_TASK_ID*3))

python3 ./DQN_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env catcher -run_number first_run
python3 ./DQN_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env catcher -run_number second_run
python3 ./DQN_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env catcher -run_number third_run
deactivate
