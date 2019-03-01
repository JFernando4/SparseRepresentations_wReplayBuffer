#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_distreg_b
#SBATCH --output=./outputs/mc_distreg_b-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ \
-reg_factor $RF -beta $BETA -run_number $SLURM_ARRAY_TASK_ID
deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
