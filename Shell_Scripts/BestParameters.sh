#!/usr/bin/env bash


usage()
{
    echo "Usage: runs the best parameter combination experiment for the given method using the specified"\
          "buffer size for a given number of runs."
    echo "Parameters:"
    echo "  Short Form:     Long Form:                  Description:"
    echo "  -m              --method                      one of the following:"
    echo "                                                    - DQN,"
    echo "                                                    - DistributionalRegularizers_Beta,"
    echo "                                                    - DistributionalRegularizers_Gamma,"
    echo "                                                    - L1_Regularization_OnWeights,"
    echo "                                                    - L1_Regularization_OnActivations,"
    echo "                                                    - L2_Regularization_OnWeights,"
    echo "                                                    - L2_Regularization_OnActivations,"
    echo "                                                    - Dropout."
    echo "  -bs             --buffer_size                 size of the experience replay buffer"
    echo "  -n              --number_of_runs              number of the last run"
    echo "  -i              --initial_run_number          number of the first run"
    echo "  -env            --environment                 mountain_car, acrobot, or puddle_world"
    echo "  -h              --help                        print this message"
}

FIRST=1
while [ "$1" != "" ]; do
    case $1 in
        -m | --method )                     shift
                                            METHOD=$1
                                            ;;
        -bs | --buffer_size )               shift
                                            BUFFER=$1
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
echo "Method: $METHOD"
echo "Buffer Size: $BUFFER"
echo "Initial Number: $FIRST, Last Number: $RUNS"

export PYTHONPATH=.
for ((i=$FIRST; i <= $RUNS; i++))
do
    echo "Working on run $i..."
    python3 ./BestParameters_Experiment.py -env $ENV -m $METHOD -buffer_size $BUFFER -v -run_number $i
done
