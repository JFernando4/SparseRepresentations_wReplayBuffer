#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-10
#SBATCH --time=5:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=catch_drg
#SBATCH --output=./outputs/catch_drg-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
RUN1=$(($SLURM_ARRAY_TASK_ID*3 - 2))
RUN2=$(($SLURM_ARRAY_TASK_ID*3 - 1))
RUN3=$(($SLURM_ARRAY_TASK_ID*3))

python3 ./DistritbutionalReg_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR \
-beta $BETA -reg_factor $RF -use_gamma -env catcher -run_number $RUN1
python3 ./DistritbutionalReg_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR \
-beta $BETA -reg_factor $RF -use_gamma -env catcher -run_number $RUN2
python3 ./DistritbutionalReg_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR \
-beta $BETA -reg_factor $RF -use_gamma -env catcher -run_number $RUN3
deactivate