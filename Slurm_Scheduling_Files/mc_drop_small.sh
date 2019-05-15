#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=mc_drop_small
#SBATCH --output=./outputs/mc_drop_small-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

# TR stands for Total Runs
for ((i=1; i<=$TR; i++))
do
    RUN_NUMBER=$(($SLURM_ARRAY_TASK_ID*$TR - $TR + $i))
    python3 ./Dropout_Experiment.py -env mountain_car -small_network -run_number $RUN_NUMBER \
     -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -drop_prob $DP
done

deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# dropout_probability = {0.1, 0.2, 0.3, 0.4, 0.5}
