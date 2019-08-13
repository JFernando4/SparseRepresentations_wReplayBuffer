#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=dropout
#SBATCH --output=./outputs/dropout-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

# TR stands for Total Runs
for ((i=1; i<=$TR; i++))
do
    RUN_NUMBER=$(($SLURM_ARRAY_TASK_ID*$TR - $TR + $i))
    python3 ./Dropout_Experiment.py -env $ENV -run_number $RUN_NUMBER -buffer_size $BUFFER -tnet_update_freq $FREQ \
     -lr $LR -drop_prob $DP -v
done

deactivate
# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025} for mountain car
# learning rate = {0.001, 0.00025, 0.0000625, 0.000015625} for catcher
# dropout_probability = {0.1, 0.2, 0.3, 0.4, 0.5}