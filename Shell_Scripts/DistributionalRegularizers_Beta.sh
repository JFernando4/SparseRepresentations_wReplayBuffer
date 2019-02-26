#!/usr/bin/env bash


usage()
{
    echo "Usage: runs the distributional regularizers experiment with a Exponential distribution using the specified"\
          "parameters and the specified number of runs."
    echo "Parameters:"
    echo "  Short Form:     Long Form:                  Description:"
    echo "  -lr             --learning_rate               learning rate for the Adam optimizer"
    echo "  -b              --beta                        max value for the beta parameter of the Exponential distribution"
    echo "  -rf             --regularization_factor       regularization factor for the loss function"
    echo "  -n              --number_of_runs              number of the last run"
    echo "  -i              --initial_run_number          number of the first run"
    echo "  -env            --environment                 mountain_car, acrobot, or puddle_world"
    echo "  -h              --help                        print this message"
}

FIRST=1
while [ "$1" != "" ]; do
    case $1 in
        -lr | --learning_rate )             shift
                                            LR=$1
                                            ;;
        -b  | --beta )                      shift
                                            BETA=$1
                                            ;;
        -rf | --regularization_factor )     shift
                                            RF=$1
                                            ;;
        -n  | --number_of_runs )            shift
                                            RUNS=$1
                                            ;;
        -i  | --initial_run_number )        shift
                                            FIRST=$1
                                            ;;
        -env | --environment )              shift
                                            ENV=$1
                                            ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

echo "Environment: $ENV"
echo "Learning Rate: $LR"
echo "Beta: $BETA"
echo "Regularization Factor $RF"
echo "Initial Number: $FIRST, Number of Runs: $RUNS"

export PYTHONPATH=.
for ((i=$FIRST; i <= $RUNS; i++))
do
    echo "Working on run $i..."
    python3 ./DistritbutionalReg_Experiment.py -lr $LR -reg_factor $RF -beta $BETA -v -run_number $i -env $ENV
done

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
# ma_alpha = {0.1, 0.01}