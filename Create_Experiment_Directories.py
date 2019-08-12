import argparse
import os

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', action='store', default='dqn', type=str,
                        choices=['dqn', 'drg', 'dre', 'l1w', 'l2w', 'dropout'])
    parser.add_argument('-env', action='store', default='mountain_car', type=str,
                        choices=['mountain_car', 'catcher'])
    parser.add_argument('-lbs', '--limit_buffer_size', action='store_true')
    parser.add_argument('-bsv', '--buffer_size_value', action='store', type=int, default=20000)
    parser.add_argument('-lf', '--limit_freq', action='store_true')
    parser.add_argument('-fv', '--freq_value', action='store', type=int, default=10)
    parser.add_argument('-bp', '--best_parameters', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_arguments = parser.parse_args()

    methods = {
        # this specifies the parameter sweep for each method:
        #   parameter_names are the parameters over which we're sweeping
        #   for each parameter name, the directory specifies the values that we're sweeping over
        'dqn': {'method': 'DQN',
                'parameter_names': ['BufferSize', 'Freq', 'LearningRate'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400]},
        'drg': {'method': 'DRG',
                'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400],
                'Beta': [0.1, 0.2, 0.5],
                'RegFactor': [0.1, 0.01, 0.001]},
        'dre': {'method': 'DRE',
                'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400],
                'Beta': [0.1, 0.2, 0.5],
                'RegFactor': [0.1, 0.01, 0.001]},
        'l1w': {'method': 'L1_Regularization_OnWeights',
                'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400],
                'RegFactor': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
        'l1a': {'method': 'L1_Regularization_OnActivations',
                'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400],
                'RegFactor': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
        'l2w': {'method': 'L2_Regularization_OnWeights',
                'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400],
                'RegFactor': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
        'l2a': {'method': 'L2_Regularization_OnActivations',
                'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'RegFactor'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                'BufferSize': [100, 1000, 5000, 20000, 80000],
                'Freq': [10, 50, 100, 200, 400],
                'RegFactor': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
        'dropout': {'method': 'Dropout',
                    'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'DropoutProbability'],
                    'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625, 0.000015625],
                    'BufferSize': [100, 1000, 5000, 20000, 80000],
                    'Freq': [10, 50, 100, 200, 400],
                    'DropoutProbability': [0.1, 0.2, 0.3, 0.4, 0.5]},
    }
    method_dictionary = methods[exp_arguments.method]

    if not exp_arguments.best_parameters:
        assert exp_arguments.buffer_size_value in method_dictionary['BufferSize']
        assert exp_arguments.freq_value in method_dictionary['Freq']

        """ General results directory """
        results_parent_directory = os.path.join(os.getcwd(), 'Results')
        if not os.path.exists(results_parent_directory):
            os.makedirs(results_parent_directory)
        """ Directory specific to the environment and the method """
        environment_result_directory = os.path.join(results_parent_directory, exp_arguments.env,
                                                    method_dictionary['method'])
        if not os.path.exists(environment_result_directory):
            os.makedirs(environment_result_directory)
        """ Directory specific to the parameters"""
        parameter_combinations = []
        for param in method_dictionary['parameter_names']:
            # parameter names are ordered
            if len(parameter_combinations) == 0:
                for parameter_value in method_dictionary[param]:    # The first parameter is always the learning rate
                    parameter_combinations.append(param + str(parameter_value))
            else:
                temp_list = []
                for combination in parameter_combinations:
                    for parameter_value in method_dictionary[param]:
                        if param == 'BufferSize' and exp_arguments.limit_buffer_size:
                            if parameter_value == exp_arguments.buffer_size_value:
                                temp_list.append(combination + '_' + param + str(parameter_value))
                        elif param == 'Freq' and exp_arguments.limit_freq:
                            if parameter_value == exp_arguments.freq_value:
                                temp_list.append(combination + '_' + param + str(parameter_value))
                        else:
                            temp_list.append(combination + '_' + param + str(parameter_value))
                parameter_combinations = temp_list

        for combination in parameter_combinations:
            parameters_result_directory = os.path.join(environment_result_directory, combination)
            if not os.path.exists(parameters_result_directory):
                if exp_arguments.verbose:
                    print("Creating directory:", parameters_result_directory)
                os.makedirs(parameters_result_directory)
            else:
                print("Directory already exists:", parameters_result_directory)

    else:
        from Experiment_Engine.util import BEST_PARAMETERS_DICTIONARY

        env_name = exp_arguments.env
        method = method_dictionary['method']
        buffer_size = exp_arguments.buffer_size_value
        """ General results directory """
        results_parent_directory = os.path.join(os.getcwd(), 'Best_Parameters_Results')
        if not os.path.exists(results_parent_directory):
            os.makedirs(results_parent_directory)
        """ Directory specific to the environment """
        environment_result_directory = os.path.join(results_parent_directory, env_name)
        if not os.path.exists(environment_result_directory):
            os.makedirs(environment_result_directory)
        """ Directory specific to the method """
        method_result_directory = os.path.join(environment_result_directory, method)
        if not os.path.exists(method_result_directory):
            os.makedirs(method_result_directory)

        """ Directory specific to the parameters of the specific method and buffer size combination """
        parameters_name = 'BufferSize' + str(buffer_size)
        for name in BEST_PARAMETERS_DICTIONARY[env_name][method]['ParameterNames'][1:]:
            parameters_name += "_" + name + str(BEST_PARAMETERS_DICTIONARY[env_name][method][buffer_size][name])
        parameters_result_directory = os.path.join(method_result_directory, parameters_name)
        if exp_arguments.verbose:
            print("Creating directory:", parameters_result_directory)
        if not os.path.exists(parameters_result_directory):
            os.makedirs(parameters_result_directory)
