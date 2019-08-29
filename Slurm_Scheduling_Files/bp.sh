#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=500M
#SBATCH --job-name=bp
#SBATCH --output=./outputs/bp-%A_%a.out

source ./bin/activate
export PYTHONPATH=.

# TR stands for Total Runs
for ((i=1; i<=$TR; i++))
do
    RUN_NUMBER=$(($SLURM_ARRAY_TASK_ID*$TR - $TR + $i))
    python3 ./BestParameters_Experiment.py -env $ENV -run_number $RUN_NUMBER -buffer_size $BUFFER -m $METHOD -v
done

deactivate
# Methods = {DQN, DRE, DRE_LB, DRG, DRG_LB, L1A, L1W, L2A, L2W, Dropout}
