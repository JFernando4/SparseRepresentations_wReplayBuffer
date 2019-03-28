import os
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch

from Experiment_Engine import ParameterCombinationSummary, compute_activation_map, TwoLayerFullyConnected, \
    compute_instance_sparsity, TwoLayerDropoutFullyConnected, compute_activation_overlap

NUMBER_OF_EPISODES = 500

BEST_PARAMETERS_DICTIONARY = {

    'DQN': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.004},
        1000: {'Freq': 10, 'LearningRate': 0.004},
        5000: {'Freq': 10, 'LearningRate': 0.004},
        20000: {'Freq': 10, 'LearningRate': 0.001},
        80000: {'Freq': 10, 'LearningRate': 0.001},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate']
    },

    'DistributionalRegularizers_Beta': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.001, 'Beta': 0.2, 'RegFactor': 0.1},
        1000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.01},
        5000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
        20000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.01},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'Beta': 0.5, 'RegFactor': 0.1},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor']
    },

    'DistributionalRegularizers_Gamma': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
        1000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
        5000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.1},
        20000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.01},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'Beta': 0.2, 'RegFactor': 0.001},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor']
    },

    'L1_Regularization_OnWeights': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.001, 'RegFactor': 0.0005},
        1000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.01},
        5000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.01},
        20000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.01},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.01},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
    },

    'L1_Regularization_OnActivations': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.00025, 'RegFactor': 0.1},
        1000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
        5000: {'Freq': 10, 'LearningRate': 0.004, 'RegFactor': 0.0001},
        20000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.001},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.001},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
    },

    'L2_Regularization_OnWeights': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.004, 'RegFactor': 0.0005},
        1000: {'Freq': 10, 'LearningRate': 0.01, 'RegFactor': 0.05},
        5000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.001},
        20000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.01},
        80000: {'Freq': 10, 'LearningRate': 0.004, 'RegFactor': 0.1},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
    },

    'L2_Regularization_OnActivations': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.001, 'RegFactor': 0.001},
        1000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.05},
        5000: {'Freq': 10, 'LearningRate': 0.00025, 'RegFactor': 0.1},
        20000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.05},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.05},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
    },

    'Dropout': {
        # Buffer Size
        100: {'Freq': 400, 'LearningRate': 0.001, 'DropoutProbability': 0.1},
        1000: {'Freq': 10, 'LearningRate': 0.001, 'DropoutProbability': 0.1},
        5000: {'Freq': 10, 'LearningRate': 0.001, 'DropoutProbability': 0.1},
        20000: {'Freq': 10, 'LearningRate': 0.001, 'DropoutProbability': 0.2},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'DropoutProbability': 0.2},
        'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'DropoutProbability']
    }
}

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--environment', action='store', default='mountain_car', type=str,
                        choices=['mountain_car', 'catcher'])
    parser.add_argument('-m', '--method', nargs='+', help='<Required> Set flag', required=True, type=str,
                        choices=['DQN', 'DistributionalRegularizers_Gamma', 'DistributionalRegularizers_Beta',
                                 'L1_Regularization_OnWeights', 'L1_Regularization_OnActivations',
                                 'L2_Regularization_OnWeights', 'L2_Regularization_OnActivations',
                                 'Dropout'])
    parser.add_argument('-bs', '--buffer_size', nargs="+", help='<Required> Set flag', required=True, type=int,
                        choices=[100, 1000, 5000, 20000, 80000])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-cs_am', '--compute_summaries_and_activation_maps', action='store_true')
    parser.add_argument('-sp', '--save_plots', action='store_true')
    arguments = parser.parse_args()
    methods = arguments.method
    buffer_sizes = arguments.buffer_size

    summary_names = ['return_per_episode', 'steps_per_episode', 'cumulative_loss_per_episode']
    perf_measure_name = 'return_per_episode'

    """ Best parameters results directory """
    results_dir = os.path.join(os.getcwd(), 'Best_Parameters_Results')
    """ Environment results directory """
    env_dir = os.path.join(results_dir, arguments.environment)

    methods_directory_paths = []
    for method_name in methods:
        """ Method results directory """
        method_results_directory = os.path.join(env_dir, method_name)
        method_parameter_names = BEST_PARAMETERS_DICTIONARY[method_name]['ParameterNames']
        for size in buffer_sizes:
            """ Parameter combination directory """
            method_parameter_combination_name = 'BufferSize' + str(size)
            method_parameter_dictionary = BEST_PARAMETERS_DICTIONARY[method_name][size]
            for name in method_parameter_names[1:]:
                method_parameter_combination_name += '_' + name + str(method_parameter_dictionary[name])
            parameter_combination_directory = os.path.join(method_results_directory, method_parameter_combination_name)
            print(parameter_combination_directory)
            if not os.path.exists(parameter_combination_directory):
                raise ValueError("Couldn't find the directory for the method: " + method_name + "," +\
                                 "and the parameter combination: " + method_parameter_combination_name)
            else:
                # If there is a directory for the given method and parameter combination, then store the parameters
                # names and the path to the directory
                methods_directory_paths.append((parameter_combination_directory, method_parameter_combination_name,
                                                method_parameter_names, method_name))

    if arguments.compute_summaries_and_activation_maps:
        for parameter_combination_directory, method_parameter_combination_name, method_parameter_names, method_name in \
                methods_directory_paths:
            method_summary = ParameterCombinationSummary(
                param_comb_path=parameter_combination_directory, param_comb_name=method_parameter_combination_name,
                parameter_names=method_parameter_names, summary_names=summary_names,
                performance_measure_name=perf_measure_name
            )

            if arguments.verbose:
                print("\n")
                print('#------------------------- Summary of method: -------------------------#')
                print("Method Name:", method_name)
                method_summary.print_summary(2)
                print('#----------------------------------------------------------------------#')
                print("\n")

            """ Computing activation maps """
            layer1_active_neurons = []
            layer1_percentage_of_active = []
            layer1_dead_neurons = np.zeros(method_summary.sample_size, dtype=np.int64)
            layer1_activation_overlap = np.zeros(method_summary.sample_size, dtype=np.float64)

            layer2_active_neurons = []
            layer2_percentage_of_active = []
            layer2_dead_neurons = np.zeros(method_summary.sample_size, dtype=np.int64)
            layer2_activation_overlap = np.zeros(method_summary.sample_size, dtype=np.float64)

            for i in range(method_summary.sample_size):
                if arguments.verbose:
                    print("Working on activation maps of run", i+1, 'of method', method_name + '...')
                network_weights_path = method_summary.runs[i]['weights_path']
                if arguments.method == 'Dropout':
                    dropout_probability = method_summary.parameter_values['DropoutProbability']
                    net = TwoLayerDropoutFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3,
                                                        gates='relu-relu', dropout_probability=dropout_probability)
                else:
                    net = TwoLayerFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3, gates='relu-relu')
                net.load_state_dict(torch.load(network_weights_path))
                net.eval()

                l1, l2 = compute_activation_map(net, 100)
                # Since the runs are ordered from lowest to highest performance, the activation maps are also in order
                layer1_active, layer1_percentage = compute_instance_sparsity(l1)
                layer1_active_neurons.append(layer1_active), layer1_percentage_of_active.append(layer1_percentage)
                l1_dead = 32 - l1.shape[0]
                layer1_dead_neurons[i] += l1_dead
                layer1_overlap = compute_activation_overlap(l1, granularity=10)
                layer1_activation_overlap[i] += layer1_overlap

                layer2_active, layer2_percentage = compute_instance_sparsity(l2)
                layer2_active_neurons.append(layer2_active), layer2_percentage_of_active.append(layer2_percentage)
                l2_dead = 256 - l2.shape[0]
                layer2_dead_neurons[i] += l2_dead
                layer2_overlap = compute_activation_overlap(l2, granularity=10)
                layer2_activation_overlap[i] += layer2_overlap
                if arguments.verbose:
                    print("\tDead neurons in layer 1:", l1_dead)
                    print("\tActivation overlap in layer 1:", np.round(layer1_overlap, 2))
                    print("\tDead neurons in layer 2:", l2_dead)
                    print("\tActivation overlap in layer 2:", np.round(layer2_overlap, 2))

            l1_avg_dead = str(np.round(np.average(layer1_dead_neurons), 2))
            l2_avg_dead = str(np.round(np.average(layer2_dead_neurons), 2))
            summary_file_path = os.path.join(parameter_combination_directory, 'summary.txt')
            method_summary.write_summary(path=summary_file_path, round_dec=2,
                                         extra_summary_lines=
                                         ('The average number of dead neurons in the first layer is: ' + l1_avg_dead,
                                          'The average number of dead neurons in the second layer is: ' + l2_avg_dead))

            method_summary_file_path = os.path.join(parameter_combination_directory, 'method_summary.p')
            with open(method_summary_file_path, mode='wb') as method_summary_file:
                pickle.dump(method_summary, method_summary_file)
            overlap_file_path = os.path.join(parameter_combination_directory, 'activation_overlap.p')
            with open(overlap_file_path, mode='wb') as overlap_file:
                activation_overlap_dictionary = {
                    'layer1_activation_overlap': layer1_activation_overlap,
                    'layer1_dead_neurons': layer1_dead_neurons,
                    'layer2_activation_overlap': layer2_activation_overlap,
                    'layer2_dead_neurons': layer2_dead_neurons
                }
                pickle.dump(activation_overlap_dictionary, overlap_file)
            instance_sparsity_file_path = os.path.join(parameter_combination_directory, 'instance_sparsity.p')
            with open(instance_sparsity_file_path, mode='wb') as instance_sparsity_file:
                instance_sparsity_dictionary = {
                    'layer1_active_neurons': layer1_active_neurons,
                    'layer1_percentage_of_active': layer1_percentage_of_active,
                    'layer2_active_neurons': layer2_active_neurons,
                    'layer2_percentage_of_active': layer2_percentage_of_active
                }
                pickle.dump(instance_sparsity_dictionary, instance_sparsity_file)

    # ###################
    # #      Plots      #
    # ###################
    # runs = method_summary.runs
    # first_idx = 0
    # last_idx = len(runs) - 1
    # median_idx = int(np.floor(last_idx / 2))
    # first_q = int(np.floor(median_idx / 2))
    # third_q = median_idx + int(np.floor((last_idx - median_idx) / 2))
    # indices = [first_idx, first_q, median_idx, third_q, last_idx]
    #
    # print('The average return of the worst run was:', np.average(runs[first_idx]['summary']['return_per_episode']))
    # print('The average return of the 25 percentile was:', np.average(runs[first_q]['summary']['return_per_episode']))
    # print('The average return of the median run was:', np.average(runs[median_idx]['summary']['return_per_episode']))
    # print('The average return of the 75 percentile was:', np.average(runs[third_q]['summary']['return_per_episode']))
    # print('The average return of the best run was:', np.average(runs[last_idx]['summary']['return_per_episode']))
    #
    # colors = [
    #     '#FFB43B',  # yellow-ish        -   worst
    #     '#DE6464',  # salmon            -   25 percentile
    #     '#AD3465',  # dark pink-ish     -   median
    #     '#601569',  # dark purple-ish   -   75 percentile
    #     '#057DB5'   # blue              -   best
    # ]
    #
    # """ Activation Maps """
    # names_of_runs = ['worst run', '25 percentile', 'median', '75 percentile', 'best run']
    #
    # layer2_active_percentage_low_performance = None
    # print('\n')
    #
    # for i, idx in enumerate(range(50)):
    #     network_weights_path = runs[idx]['weights_path']
    #     if arguments.method == 'Dropout':
    #         dropout_probability = method_summary.parameter_values['DropoutProbability']
    #         net = TwoLayerDropoutFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3,
    #                                             gates='relu-relu', dropout_probability=dropout_probability)
    #     else:
    #         net = TwoLayerFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3, gates='relu-relu')
    #     net.load_state_dict(torch.load(network_weights_path))
    #     net.eval()
    #
    #     l1, l2 = compute_activation_map(net, 100)
    #     print('Dead neurons of run ' + str(idx+1) + ':', '\tlayer 1:', 32 - l1.shape[0], '\tlayer 2:',
    #           256 - l2.shape[0])
    #
    #     layer2_active, layer2_percentage = compute_instance_sparsity(l2)
    #     if layer2_active_percentage_low_performance is None:
    #         layer2_active_percentage_low_performance = layer2_percentage
    #     else:
    #         layer2_active_percentage_low_performance = np.append(layer2_active_percentage_low_performance,
    #                                                              layer2_percentage)
    #
    # layer2_active_percentage_high_performance = None
    # print('\n')