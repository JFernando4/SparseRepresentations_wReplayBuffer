import argparse
import os

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', action='store', default='dqn', type=str,
                        choices=['dqn', 'dist_reg_gamma', 'dist_reg_beta'])
    parser.add_argument('-env', action='store', default='acrobot', type=str,
                        choices=['mountain_car', 'acrobot', 'puddle_world'])
    parser.add_argument('-lbs', '--limit_buffer_size', action='store_true')
    parser.add_argument('-bsv', '--buffer_size_value', action='store', type=int, default=20000)
    parser.add_argument('-lf', '--limit_freq', action='store_true')
    parser.add_argument('-fv', '--freq_value', action='store', type=int, default=10)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_arguments = parser.parse_args()

    methods = {
        # this specifies the parameter sweep for each method:
        #   parameter_names are the parameters over which we're sweeping
        #   for each parameter name, the directory specifies the values that we're sweeping over
        'dqn': {'method': 'DQN', 'parameter_names': ['LearningRate', 'BufferSize', 'Freq'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025],
                'BufferSize': [100, 1000, 5000, 10000, 20000, 40000],
                'Freq': [10, 50, 100, 200, 400]},
        'dist_reg_gamma': {'method': 'DistributionalRegularizers_Gamma',
                           'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
                           'LearningRate': [0.01, 0.004, 0.001, 0.00025],
                           'BufferSize': [100, 1000, 5000, 20000, 40000],
                           'Freq': [10, 50, 100, 200, 400],
                           'Beta': [0.1, 0.2, 0.5],
                           'RegFactor': [0.1, 0.01, 0.001]},
        'dist_reg_beta': {'method': 'DistributionalRegularizers_Beta',
                          'parameter_names': ['LearningRate', 'BufferSize', 'Freq', 'Beta', 'RegFactor'],
                          'LearningRate': [0.01, 0.004, 0.001, 0.00025],
                          'BufferSize': [100, 1000, 5000, 20000, 40000],
                          'Freq': [10, 50, 100, 200, 400],
                          'Beta': [0.1, 0.2, 0.5],
                          'RegFactor': [0.1, 0.01, 0.001]},
    }
    method_dictionary = methods[exp_arguments.method]
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
            for parameter_value in method_dictionary[param]:
                if param == 'BufferSize' or param == 'Freq':
                    if exp_arguments.limit_buffer_size or exp_arguments.limit_freq:
                        if (parameter_value == exp_arguments.buffer_size_value) or \
                           (parameter_value == exp_arguments.freq_value):
                                parameter_combinations.append(param + str(parameter_value))
                    else:
                        parameter_combinations.append(param + str(parameter_value))
                else:
                    parameter_combinations.append(param + str(parameter_value))
        else:
            temp_list = []
            for combination in parameter_combinations:
                for parameter_value in method_dictionary[param]:
                    if param == 'BufferSize' or param == 'Freq':
                        if exp_arguments.limit_buffer_size or exp_arguments.limit_freq:
                            if (parameter_value == exp_arguments.buffer_size_value) or \
                               (parameter_value == exp_arguments.freq_value):
                                    temp_list.append(combination + '_' + param + str(parameter_value))
                        else:
                            temp_list.append(combination + '_' + param + str(parameter_value))
                    else:
                        temp_list.append(combination + '_' + param + str(parameter_value))
            parameter_combinations = temp_list

    for combination in parameter_combinations:
        parameters_result_directory = os.path.join(environment_result_directory, combination)
        if exp_arguments.verbose:
            print("Creating directory:", parameters_result_directory)
        if not os.path.exists(parameters_result_directory):
            os.makedirs(parameters_result_directory)
