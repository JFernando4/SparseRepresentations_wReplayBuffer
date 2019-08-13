#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=l1a
#SBATCH --output=./outputs/l1a-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

# TR stands for Total Runs
for ((i=1; i<=$TR; i++))
do
    RUN_NUMBER=$(($SLURM_ARRAY_TASK_ID*$TR - $TR + $i))
    python3 ./Regularization_Experiment.py -env $ENV -run_number $RUN_NUMBER -buffer_size $BUFFER -tnet_update_freq $FREQ -v \
     -l1_reg -lr $LR -reg_factor $RF
done

deactivate
# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025} for mountain car
# learning rate = {0.001, 0.00025, 0.0000625, 0.000015625} for catcher
# reg_factor = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
