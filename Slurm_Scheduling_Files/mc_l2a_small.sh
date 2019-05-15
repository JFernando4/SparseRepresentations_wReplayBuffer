#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=mc_l2a_small
#SBATCH --output=./outputs/mc_l2a_small-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

# TR stands for Total Runs
for ((i=1; i<=$TR; i++))
do
    RUN_NUMBER=$(($SLURM_ARRAY_TASK_ID*$TR - $TR + $i))
    python3 ./Regularization_Experiment.py -env mountain_car -small_network -run_number $RUN_NUMBER  \
     -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -reg_factor $RF
done


# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# reg_factor = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
