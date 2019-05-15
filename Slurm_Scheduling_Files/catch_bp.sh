#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --array=1-100
#SBATCH --time=10:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=catch_bp
#SBATCH --output=./outputs/catch_bp-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

RUN1=$(($SLURM_ARRAY_TASK_ID*5 - 4))
RUN2=$(($SLURM_ARRAY_TASK_ID*5 - 3))
RUN3=$(($SLURM_ARRAY_TASK_ID*5 - 2))
RUN4=$(($SLURM_ARRAY_TASK_ID*5 - 1))
RUN5=$(($SLURM_ARRAY_TASK_ID*5))

python3 ./BestParameters_Experiment.py -v -env catcher -buffer_size $BUFFER -m $METHOD -run_number $RUN1
python3 ./BestParameters_Experiment.py -v -env catcher -buffer_size $BUFFER -m $METHOD -run_number $RUN2
python3 ./BestParameters_Experiment.py -v -env catcher -buffer_size $BUFFER -m $METHOD -run_number $RUN3
python3 ./BestParameters_Experiment.py -v -env catcher -buffer_size $BUFFER -m $METHOD -run_number $RUN4
python3 ./BestParameters_Experiment.py -v -env catcher -buffer_size $BUFFER -m $METHOD -run_number $RUN5
deactivate
