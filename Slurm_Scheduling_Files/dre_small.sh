#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=dre_small
#SBATCH --output=./outputs/dre_small-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

# TR stands for Total Runs
for ((i=1; i<=$TR; i++))
do
    RUN_NUMBER=$(($SLURM_ARRAY_TASK_ID*$TR - $TR + $i))
    python3 ./DistritbutionalReg_Experiment.py -env $ENV -small_network -run_number $RUN_NUMBER \
     -lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ -reg_factor $RF -beta $BETA
done

deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025} for mountain car
# learning rate = {0.001, 0.00025, 0.0000625, 0.000015625} for catcher
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
