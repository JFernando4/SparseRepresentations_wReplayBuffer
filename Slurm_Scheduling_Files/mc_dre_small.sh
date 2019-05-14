#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=mc_dre_small
#SBATCH --output=./outputs/mc_dre_small-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

RUN1=$(($SLURM_ARRAY_TASK_ID*5 - 4))
RUN2=$(($SLURM_ARRAY_TASK_ID*5 - 3))
RUN3=$(($SLURM_ARRAY_TASK_ID*5 - 2))
RUN4=$(($SLURM_ARRAY_TASK_ID*5 - 1))
RUN5=$(($SLURM_ARRAY_TASK_ID*5))

python3 ./DistritbutionalReg_Experiment.py -env mountain_car -small_network -run_number $RUN1 \
-lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ -reg_factor $RF -beta $BETA
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -small_network -run_number $RUN2 \
-lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ -reg_factor $RF -beta $BETA
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -small_network -run_number $RUN3 \
-lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ -reg_factor $RF -beta $BETA
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -small_network -run_number $RUN4 \
-lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ -reg_factor $RF -beta $BETA
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -small_network -run_number $RUN5 \
-lr $LR -buffer_size $BUFFER -tnet_update_freq $FREQ -reg_factor $RF -beta $BETA

deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
