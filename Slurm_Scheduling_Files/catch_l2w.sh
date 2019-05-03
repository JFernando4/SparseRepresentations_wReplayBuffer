#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-10
#SBATCH --time=5:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=catch_l2w
#SBATCH --output=./outputs/catch_l2w-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
RUN1=$(($SLURM_ARRAY_TASK_ID*3 - 2))
RUN2=$(($SLURM_ARRAY_TASK_ID*3 - 1))
RUN3=$(($SLURM_ARRAY_TASK_ID*3))

python3 ./Regularization_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR \
-reg_factor $RF -env catcher -weights_reg -run_number $RUN1
python3 ./Regularization_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR \
-reg_factor $RF -env catcher -weights_reg -run_number $RUN2
python3 ./Regularization_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR \
-reg_factor $RF -env catcher -weights_reg -run_number $RUN3
deactivate
