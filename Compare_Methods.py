import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from Experiment_Engine import ParameterCombinationSummary, compare_sample_average

NUMBER_OF_EPISODES = 500


def parse_method_parameters(parameter_names, parameter_values):
    assert len(parameter_names) == len(parameter_values)
    parameter_combination_name = ''
    for i in range(len(parameter_values) - 1):
        parameter_combination_name += parameter_names[i] + str(parameter_values[i]) + "_"
    parameter_combination_name += parameter_names[-1] + str(parameter_values[-1])
    return parameter_combination_name


def get_method_results_directory(environment_name, method_name):
    method_results_directory = os.path.join(os.getcwd(), 'Results', environment_name, method_name)
    if not os.path.isdir(method_results_directory):
        raise ValueError("There are no result for that combination of environment and method.")
    return method_results_directory


if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', action='store', default='mountain_car', type=str, choices=['mountain_car', 'acrobot',
                                                                                           'puddle_world'])
    parser.add_argument('-m1', '--method1_name', action='store', default='DQN', type=str,
                        choices=['DQN', 'DistributionalRegularizers_Gamma', 'DistributionalRegularizers_Beta'])
    parser.add_argument('-m1p', '--method1_parameters', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument('-m2', '--method2_name', action='store', default='DQN', type=str,
                        choices=['DQN', 'DistributionalRegularizers_Gamma', 'DistributionalRegularizers_Beta'])
    parser.add_argument('-m2p', '--method2_parameters', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument('-verbose', action='store_true')
    parser.add_argument('-store_summary', action='store_true')
    parser.add_argument('-load_summary', action='store_true')
    arguments = parser.parse_args()

    parameters_dict = {
        'DQN': ['LearningRate', 'BufferSize', 'Freq'],
        'DistributionalRegularizers_Gamma': ['LearningRate', 'Beta', 'RegFactor'],
        'DistributionalRegularizers_Beta': ['LearningRate', 'Beta', 'RegFactor']
    }
    summary_names = ['return_per_episode', 'steps_per_episode', 'cumulative_loss_per_episode']
    perf_measure_name = 'return_per_episode'

    """ Method 1 results """
    method1_parameter_names = parameters_dict[arguments.method1_name]
    method1_parameter_combination = parse_method_parameters(parameter_names=method1_parameter_names,
                                                            parameter_values=arguments.method1_parameters)
    method1_results_directory = get_method_results_directory(arguments.env, arguments.method1_name)
    method1_summary = ParameterCombinationSummary(
        param_comb_path=os.path.join(method1_results_directory, method1_parameter_combination),
        param_comb_name=method1_parameter_combination, parameter_names=method1_parameter_names,
        summary_names=summary_names, performance_measure_name=perf_measure_name
    )

    """ Method 2 results """
    method2_parameter_names = parameters_dict[arguments.method2_name]
    method2_parameter_combination = parse_method_parameters(parameter_names=method2_parameter_names,
                                                            parameter_values=arguments.method2_parameters)
    method2_results_directory = get_method_results_directory(arguments.env, arguments.method2_name)
    method2_summary = ParameterCombinationSummary(
        param_comb_path=os.path.join(method2_results_directory, method2_parameter_combination),
        param_comb_name=method2_parameter_combination, parameter_names=method2_parameter_names,
        summary_names=summary_names, performance_measure_name=perf_measure_name
    )

    print("\n")
    print('#------------------------ Summary of method 1: ------------------------#')
    print("Method Name:", arguments.method1_name)
    method1_summary.print_summary(2)
    print('#----------------------------------------------------------------------#')
    print("\n")
    print('#------------------------ Summary of method 2: ------------------------#')
    print("Method Name:", arguments.method2_name)
    method2_summary.print_summary(2)
    print('#----------------------------------------------------------------------#')
    print("\n")

    compare_sample_average(method1_summary, method2_summary)

    x = np.arange(1, 501)

    method1_mean = np.average(method1_summary.summaries['return_per_episode'], axis=0)
    method1_stddev = np.std(method1_summary.summaries['return_per_episode'], ddof=1, axis=0)
    method2_mean = np.average(method2_summary.summaries['return_per_episode'], axis=0)
    method2_stddev = np.std(method2_summary.summaries['return_per_episode'], ddof=1, axis=0)

    plt.fill_between(x, method2_mean - method2_stddev, method2_mean + method2_stddev, color='#b8d1de')
    plt.plot(x, method1_mean, color='#FBB829')
    plt.fill_between(x, method1_mean - method1_stddev, method1_mean + method1_stddev, color='#fff2d5')
    plt.plot(x, method2_mean, color='#025D8C')


    plt.ylim((-750, -100))
    plt.show()
