#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --time=4:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_er
#SBATCH --output=./outputs/mc_er-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./DQN_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env mountain_car -run_number $SLURM_ARRAY_TASK_ID
deactivate