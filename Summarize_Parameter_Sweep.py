import os
import argparse
import numpy as np

from Experiment_Engine import ParameterCombinationSummary, MethodResults, extract_method_parameter_values

ENVIRONMENT_DICTIONARY = {'mountain_car': {'summary_size': 200000,
                                           'performance_measure_name': 'reward_per_step',
                                           'summary_function': np.sum,
                                           },
                          'catcher': {'summary_size': 500000,
                                      'performance_measure_name': 'reward_per_step',
                                      'summary_function': np.sum,
                                      }
                          }

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', action='store', default='mountain_car', type=str,
                        choices=['mountain_car', 'catcher'])
    parser.add_argument('-method', action='store', default='DQN', type=str,
                        choices=['DQN',         # Original DQN
                                 'DRG',         # DQN with distributional regularizers with gamma distribution
                                 'DRE',         # DQN with distributional regularizers with exponential distribution
                                 'L1W',         # DQN with L1 regularization on the weights of the representation
                                 'L1A',         # DQN with L1 regularization on the last activations the network
                                 'L2W',         # DQN with l2 regularization on the weights of the representation
                                 'L2A',         # DQN with l2 regularization on the last activations of the network
                                 'Dropout']     # DQN with dropout on the the layer of the representation
                        )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-lbs', '--limit_buffer_size', action='store_true')
    parser.add_argument('-bsv', '--buffer_size_value', action='store', type=int, default=20000)
    parser.add_argument('-ls', '--load_summary', action='store_true')
    parser.add_argument('-pbp', '--print_best_parameters', action='store_true')
    parser.add_argument('-rtr', '--refine_top_results', action='store_true')
    parser.add_argument('-pbv', '--print_bash_variables', action='store_true')
    arguments = parser.parse_args()

    parameters_dict = {
        'DQN': ['BufferSize', 'Freq', 'LearningRate'],
        'DRG': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor'],
        'DRE': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor'],
        'L1W': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor'],
        'L1A': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor'],
        'L2W': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor'],
        'L2A': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor'],
        'Dropout': ['BufferSize', 'Freq', 'LearningRate', 'DropoutProbability'],
    }

    """ Method results directory """
    method_results_directory = os.path.join(os.getcwd(), 'Results', arguments.env, arguments.method)
    if not os.path.isdir(method_results_directory):
        raise ValueError("There are no result for that combination of environment and method.")

    overall_results = MethodResults(arguments.method)
    params_name = parameters_dict[arguments.method]

    for param_comb in os.listdir(method_results_directory):
        if arguments.limit_buffer_size:
            param_names_and_values = extract_method_parameter_values(params_name, param_comb)
            if int(param_names_and_values['BufferSize']) == arguments.buffer_size_value:
                param_comb_summary = ParameterCombinationSummary(
                    param_comb_path=os.path.join(method_results_directory, param_comb),
                    param_comb_name=param_comb, parameter_names=params_name,
                    performance_measure_name=ENVIRONMENT_DICTIONARY[arguments.env]['performance_measure_name'],
                    load_summary=arguments.load_summary,
                    summary_size=ENVIRONMENT_DICTIONARY[arguments.env]['summary_size'],
                    summary_function=ENVIRONMENT_DICTIONARY[arguments.env]['summary_function'],
                )
                if arguments.verbose:
                    param_comb_summary.print_summary(2)
                    print('\n')
                overall_results.append(param_comb_summary)
        else:
            param_comb_summary = ParameterCombinationSummary(
                param_comb_path=os.path.join(method_results_directory, param_comb),
                param_comb_name=param_comb, parameter_names=params_name,
                performance_measure_name=ENVIRONMENT_DICTIONARY[arguments.env]['performance_measure_name'],
                load_summary=arguments.load_summary,
                summary_size=ENVIRONMENT_DICTIONARY[arguments.env]['summary_size'],
                summary_function=ENVIRONMENT_DICTIONARY[arguments.env]['summary_function'],
            )
            if arguments.verbose:
                param_comb_summary.print_summary(2)
                print('\n')
            overall_results.append(param_comb_summary)

    # overall_results.refine_top_results()
    print('\n\n### Top parameter combinations: ###')
    overall_results.print_top_results(arguments.print_bash_variables)
    if arguments.refine_top_results:
        overall_results.refine_top_results()
        print("### Refined Results: ###")
        overall_results.print_top_results()
    if arguments.print_best_parameters:
        overall_results.print_best_param_comb()
