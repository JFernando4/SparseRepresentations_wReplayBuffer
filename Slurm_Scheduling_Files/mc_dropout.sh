#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=mc_dropout
#SBATCH --output=./outputs/mc_dropout-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./Dropout_Experiment.py -env mountain_car -buffer_size $BUFFER -tnet_update_freq $FREQ \
-lr $LR -drop_prob $DP -run_number $SLURM_ARRAY_TASK_ID
deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# dropout_probability = {0.1, 0.2, 0.3, 0.4, 0.5}
