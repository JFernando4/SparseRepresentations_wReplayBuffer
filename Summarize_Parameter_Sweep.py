import os
import argparse

from Experiment_Engine import ParameterCombinationSummary, MethodResults, extract_method_parameter_values

ENVIRONMENT_DICTIONARY = {'mountain_car': {'summary_size': 500, 'summary_function': 'avg', 'lite': False,
                                           'summary_names': ['return_per_episode', 'steps_per_episode',
                                                             'cumulative_loss_per_episode'],
                                           'performance_measure_name': 'return_per_episode'},
                          'catcher': {'summary_size': 500000, 'summary_function': 'sum', 'lite': True,
                                      'summary_names': ['reward_per_step'],
                                      'performance_measure_name': 'reward_per_step'}
                          }

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', action='store', default='mountain_car', type=str,
                        choices=['mountain_car', 'catcher'])
    parser.add_argument('-method', action='store', default='DQN', type=str,
                        choices=['DQN',
                                 'DistributionalRegularizers_Gamma', 'DistributionalRegularizers_Gamma_OnlyLayer2',
                                 'DistributionalRegularizers_Beta', 'DistributionalRegularizers_Beta_OnlyLayer2',
                                 'L1_Regularization_OnWeights',
                                 'L1_Regularization_OnActivations', 'L1_Regularization_OnActivations_OnlyLayer2',
                                 'L2_Regularization_OnWeights',
                                 'L2_Regularization_OnActivations', 'L2_Regularization_OnActivations_OnlyLayer2',
                                 'Dropout', 'Dropout_OnlyLayer2'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-lbs', '--limit_buffer_size', action='store_true')
    parser.add_argument('-bsv', '--buffer_size_value', action='store', type=int, default=20000)
    parser.add_argument('-ls', '--load_summary', action='store_true')
    parser.add_argument('-pbp', '--print_best_parameters', action='store_true')
    parser.add_argument('-rtr', '--refine_top_results', action='store_true')
    arguments = parser.parse_args()

    parameters_dict = {
        'DQN': ['LearningRate', 'BufferSize', 'Freq'],
        'DistributionalRegularizers_Gamma': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
        'DistributionalRegularizers_Gamma_OnlyLayer2': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
        'DistributionalRegularizers_Beta': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
        'DistributionalRegularizers_Beta_OnlyLayer2': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
        'L1_Regularization_OnWeights': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
        'L1_Regularization_OnActivations': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
        'L1_Regularization_OnActivations_OnlyLayer2': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
        'L2_Regularization_OnWeights': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
        'L2_Regularization_OnActivations': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
        'L2_Regularization_OnActivations_OnlyLayer2': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
        'Dropout': ['LearningRate', 'BufferSize', 'Freq', 'DropoutProbability'],
        'Dropout_OnlyLayer2': ['LearningRate', 'BufferSize', 'Freq', 'DropoutProbability']
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
                    load_summary=arguments.load_summary
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
                load_summary=arguments.load_summary
            )
            if arguments.verbose:
                param_comb_summary.print_summary(2)
                print('\n')
            overall_results.append(param_comb_summary)

    # overall_results.refine_top_results()
    print('\n\n### Top parameter combinations: ###')
    overall_results.print_top_results()
    if arguments.refine_top_results:
        overall_results.refine_top_results()
        print("### Refined Results: ###")
        overall_results.print_top_results()
    if arguments.print_best_parameters:
        overall_results.print_best_param_comb()