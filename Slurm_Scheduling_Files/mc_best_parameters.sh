#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-500
#SBATCH --time=4:00:00
#SBATCH --account=def-sutton
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=mc_bp
#SBATCH --output=./outputs/mc_bp-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./BestParameters_Experiment.py -v -env mountain_car -buffer_size $BUFFER -m $METHOD -run_number $SLURM_ARRAY_TASK_ID
deactivate
