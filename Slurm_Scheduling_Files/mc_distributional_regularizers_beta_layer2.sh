#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --time=4:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=mc_distb_layer2
#SBATCH --output=./outputs/mc_distb_layer2-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ \
-reg_factor $RF -beta $BETA -layer2 -run_number $SLURM_ARRAY_TASK_ID
deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
