#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=mc_l1reg
#SBATCH --output=./outputs/mc_l1reg-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./Regularization_Experiment.py -env mountain_car -buffer_size $BUFFER -tnet_update_freq $FREQ \
-lr $LR -reg_factor $RF -l1_reg -run_number $SLURM_ARRAY_TASK_ID
deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# reg_factor = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
