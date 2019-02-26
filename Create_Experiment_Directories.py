import argparse
import os

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-method_name', action='store', default='dqn', type=str,
                        choices=['dqn', 'l1_reg', 'l2_reg', 'dist_reg_gamma', 'dist_reg_beta', 'sigmoid_weighted',
                                 'emecs'])
    parser.add_argument('-env', action='store', default='acrobot', type=str, choices=['mountain_car', 'acrobot',
                                                                                           'puddle_world'])
    parser.add_argument('-verbose', action='store_true')
    exp_arguments = parser.parse_args()

    methods = {
        # this specifies the parameter sweep for each method:
        #   parameter_names are the parameters over which we're sweeping
        #   for each parameter name, the directory specifies the values that we're sweeping over
        'dqn': {'method_name': 'DQN', 'parameter_names': ['LearningRate', 'BufferSize', 'Freq'],
                'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                'BufferSize': [100, 500, 1000, 5000, 10000, 20000, 40000],
                'Freq': [10, 50, 100, 200, 400]},
        'l1_reg': {'method_name': 'L1_Regularization',
                   'parameter_names': ['LearningRate', 'RegFactor'],
                   'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                   'RegFactor': [0.1, 0.01, 0.001]},
        'l2_reg': {'method_name': 'L2_Regularization',
                   'parameter_names': ['LearningRate', 'RegFactor'],
                   'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                   'RegFactor': [0.1, 0.01, 0.001]},
        'dist_reg_gamma': {'method_name': 'DistributionalRegularizers_Gamma',
                           'parameter_names': ['LearningRate', 'Beta', 'RegFactor'],
                           'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                           'Beta': [0.1, 0.2, 0.5],
                           'RegFactor': [0.1, 0.01, 0.001]},
        'dist_reg_beta': {'method_name': 'DistributionalRegularizers_Beta',
                          'parameter_names': ['LearningRate', 'Beta', 'RegFactor'],
                          'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                          'Beta': [0.1, 0.2, 0.5],
                          'RegFactor': [0.1, 0.01, 0.001]},
        'sigmoid_weighted': {'method_name': 'SigmoidWeighted_Units',
                             'parameter_names': ['LearningRate', 'Architecture'],
                             'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                             'Architecture': ['SS', 'SD', 'DS', 'DD']},
        'emecs': {'method_name': 'Lift_And_Project',
                  'parameter_names': ['LearningRate', 'Radius'],
                  'LearningRate': [0.01, 0.004, 0.001, 0.00025, 0.0000625],
                  'Radius': [1, 2, 4, 8, 10]}
    }
    method_dictionary = methods[exp_arguments.method_name]

    """ General results directory """
    results_parent_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(results_parent_directory):
        os.makedirs(results_parent_directory)
    """ Directory specific to the environment and the method """
    environment_result_directory = os.path.join(results_parent_directory, exp_arguments.env,
                                                method_dictionary['method_name'])
    if not os.path.exists(environment_result_directory):
        os.makedirs(environment_result_directory)
    """ Directory specific to the parameters"""
    parameter_combinations = []
    for param in method_dictionary['parameter_names']:
        # parameter names are ordered
        if len(parameter_combinations) == 0:
            for parameter_value in method_dictionary[param]:
                parameter_combinations.append(param + str(parameter_value))
        else:
            temp_list = []
            for combination in parameter_combinations:
                for parameter_value in method_dictionary[param]:
                    temp_list.append(combination + '_' + param + str(parameter_value))
            parameter_combinations = temp_list

    for combination in parameter_combinations:
        parameters_result_directory = os.path.join(environment_result_directory, combination)
        if exp_arguments.verbose:
            print("Creating directory:", parameters_result_directory)
        if not os.path.exists(parameters_result_directory):
            os.makedirs(parameters_result_directory)
