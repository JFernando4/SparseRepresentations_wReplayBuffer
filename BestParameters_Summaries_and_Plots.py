import os
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch

from Experiment_Engine import ParameterCombinationSummary, compute_activation_map2D, TwoLayerFullyConnected, \
    compute_instance_sparsity, TwoLayerDropoutFullyConnected, compute_activation_overlap, sample_activation_maps, \
    compute_tdist_confidence_interval

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
        100: {'Freq': 400, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.01},
        1000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
        5000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
        20000: {'Freq': 10, 'LearningRate': 0.001, 'Beta': 0.5, 'RegFactor': 0.1},
        80000: {'Freq': 10, 'LearningRate': 0.001, 'Beta': 0.2, 'RegFactor': 0.1},
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
        1000: {'Freq': 10, 'LearningRate': 0.004, 'RegFactor': 0.001},
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
    parser.add_argument('-cs', '--compute_summaries', action='store_true',
                        help="Computes the average return, the instance sparsity, and the activation overlap of the" +\
                        "corresponding method, and saves them in separate files.")
    parser.add_argument('-amp', '--activation_map_plot', action='store_true',
                        help="Plots the activation map for the corresponding method.")
    parser.add_argument('-aos', '--activation_overlap_summary', action='store_true',
                        help="Prints the summary of the activation overlap for the given method.")
    parser.add_argument('-isp', '--instance_sparsity_plot', action='store_true',
                        help="Prints the summary of the activation overlap for the given method.")
    parser.add_argument('-isp_sep', '--instance_sparsity_separate_plots', action='store_true',
                        help='Whether to plot each method separately.')
    parser.add_argument('-psp', '--performance_summary_and_plot', action='store_true',
                        help="Prints the summary of the measure of performance and plots the learning curve.")
    parser.add_argument('-psp_se', '--performance_summary_and_plot_use_standard_error', action='store_true',
                        help="Indicates whether to use standard error or the margin of error for the shaded area of" +
                             "the plot.")
    parser.add_argument('-bsrp', '--buffer_size_results_plot', action='store_true',
                        help="Prints the summary of each method for each different buffer size and plots the results.")
    parser.add_argument('-sp', '--save_plots', action='store_true')
    arguments = parser.parse_args()
    methods = arguments.method
    buffer_sizes = arguments.buffer_size

    if arguments.compute_summaries or arguments.activation_map_plot:
        if len(arguments.method) > 1:
            raise ValueError("Provide only one method for this type of summary.")

    colors = [
        "#999999",  # gray
        "#AF4F13",  # brown
        "#7FAF1B",  # green
        "#05DAFE",  # cyan
        "#FF0033",  # red
        "#2A8FBD",  # blue
        "#FBB829",  # yellow
        "#9061C2",  # purple
    ]
    lighter_colors = [
        "#e6e6e6",  # gray
        "#ead3c4",  # brown
        "#e0ebc7",  # green
        "#ccf6fc",  # cyan
        "#ffccd6",  # red
        "#c6dfec",  # blue
        "#fff0cc",  # yellow
        "#d9c8ea",  # purple
    ]

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

    if arguments.compute_summaries:
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
                if 'Dropout' in arguments.method:
                    dropout_probability = np.float64(method_summary.parameter_values['DropoutProbability'])
                    net = TwoLayerDropoutFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3,
                                                        gates='relu-relu', dropout_probability=dropout_probability)
                else:
                    net = TwoLayerFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3, gates='relu-relu')
                net.load_state_dict(torch.load(network_weights_path))
                net.eval()

                l1, l2 = compute_activation_map2D(net, 100)
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

    elif arguments.activation_map_plot:
        for parameter_combination_directory, method_parameter_combination_name, method_parameter_names, method_name in \
                methods_directory_paths:
            method_summary_file_path = os.path.join(parameter_combination_directory, 'method_summary.p')
            with open(method_summary_file_path, mode='rb') as method_summary_file:
                method_summary = pickle.load(method_summary_file)
            assert isinstance(method_summary, ParameterCombinationSummary)

            runs = method_summary.runs
            sample_size = method_summary.sample_size
            fifteen = int(15 * sample_size / 100)
            thirty = int(30 * sample_size / 100)
            fifty = int(50 * sample_size / 100)
            seventy = int(70 * sample_size / 100)
            eighty_five = int(85 * sample_size / 100)
            indices = [fifteen, thirty, fifty, seventy, eighty_five]

            if arguments.verbose:
                print("#------------------------- Method Name: " + method_name + "-------------------------#")
                print('The average return of the 15th percentile was:',
                      np.average(runs[fifteen]['summary']['return_per_episode']))
                print('The average return of the 30th percentile was:',
                      np.average(runs[thirty]['summary']['return_per_episode']))
                print('The average return of the median run was:',
                      np.average(runs[fifty]['summary']['return_per_episode']))
                print('The average return of the 70th percentile was:',
                      np.average(runs[seventy]['summary']['return_per_episode']))
                print('The average return of the 85th run was:',
                      np.average(runs[eighty_five]['summary']['return_per_episode']))

            """ Activation Maps """
            layer1_maps = []
            layer2_maps = []
            layers = [layer1_maps, layer2_maps]

            print('\n')
            for idx in indices:
                network_weights_path = runs[idx]['weights_path']
                if 'Dropout' in arguments.method:
                    dropout_probability = np.float64(method_summary.parameter_values['DropoutProbability'])
                    net = TwoLayerDropoutFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3,
                                                        gates='relu-relu', dropout_probability=dropout_probability)
                else:
                    net = TwoLayerFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3,
                                                 gates='relu-relu')
                net.load_state_dict(torch.load(network_weights_path))
                net.eval()

                l1, l2 = compute_activation_map2D(net, 100)
                layer1_maps.extend(sample_activation_maps(l1, 5))
                layer2_maps.extend(sample_activation_maps(l2, 5))

            " Heat Maps"
            Nr = 5
            Nc = 5

            heatmap_names = ['_Layer1_Heatmap', '_Layer2_Heatmap']
            for k, layer_map in enumerate(layers):
                fig, axs = plt.subplots(Nr, Nc, figsize=(10, 10))
                images = []
                for i in range(Nr):
                    for j in range(Nc):
                        if layer_map[i * Nc + j].max() == 0:
                            data = layer_map[i * Nc + j]
                        else:
                            data = layer_map[i * Nc + j] / layer_map[i * Nc + j].max()
                        images.append(axs[i, j].imshow(data))
                        # axs[i, j].label_outer()
                        axs[i, j].axis('off')

                # [left, bottom, width, height]
                cbaxes = fig.add_axes([0.075, 0.05, 0.85, 0.03])
                images[0].set_clim(0.0, 1.0)
                cb = plt.colorbar(images[0], cax=cbaxes, orientation='horizontal')
                plt.savefig(os.path.join(parameter_combination_directory,method_name + "_" + heatmap_names[k] + '.png'))
                plt.close()

    elif arguments.activation_overlap_summary:
        for parameter_combination_directory, method_parameter_combination_name, method_parameter_names, method_name in \
                methods_directory_paths:
            activation_overlap_file_path = os.path.join(parameter_combination_directory, 'activation_overlap.p')
            with open(activation_overlap_file_path, mode='rb') as activation_overlap_file:
                activation_overlap_dictionary = pickle.load(activation_overlap_file)
            assert isinstance(activation_overlap_dictionary, dict)
            layer2_activation_overlap = activation_overlap_dictionary['layer2_activation_overlap']
            layer2_dead_neurons = activation_overlap_dictionary['layer2_dead_neurons']

            print("#------------------------- Method Name: " + method_name + "-------------------------#")
            print("Results for layer 2:")
            r = lambda z: np.round(z, 2)
            r3 = lambda z: np.round(z, 3)
            sample_size = layer2_activation_overlap.size
            avg_overlap = np.average(layer2_activation_overlap)
            stddev_overlap = np.std(layer2_activation_overlap, ddof=1)
            uci, lci, me = compute_tdist_confidence_interval(avg_overlap, stddev_overlap, 0.05, sample_size)
            print('\tAverage activation overlap:', r(avg_overlap))
            print('\tStandard deviation:', r(stddev_overlap))
            print('\t95% margin of error:', r(me),"\n")

            alive_neurons = 256 - layer2_dead_neurons
            avg_alive_neurons = np.average(alive_neurons)
            stddev_alive_neurons = np.std(alive_neurons, ddof=1)
            uci_alive, lci_alive, me_alive = compute_tdist_confidence_interval(avg_alive_neurons, stddev_alive_neurons,
                                                                               0.05, sample_size)
            print('\tAverage number of alive neurons:', r(avg_alive_neurons))
            print('\tStandard deviation:', r(stddev_alive_neurons))
            print('\t95% margin of error:', r(me_alive), '\n')

            with np.errstate(divide='ignore', invalid='ignore'):
                normed_overlap = np.divide(layer2_activation_overlap, alive_neurons)
                normed_overlap[np.isnan(normed_overlap)] = 0    # nans correspond to 0 / 0
            avg_normed_overlap = np.average(normed_overlap)
            stddev_normed_overlap = np.std(normed_overlap, ddof=1)
            uci_normed, lci_normed, me_normed = compute_tdist_confidence_interval(avg_normed_overlap,
                                                                                  stddev_normed_overlap,
                                                                                  0.05, sample_size)
            print('\tAverage normed overlap:', r(avg_normed_overlap))
            print('\tStandard deviation:', r(stddev_normed_overlap))
            print('\t95% margin of error:', r3(me_normed))

    elif arguments.instance_sparsity_plot:
        i = 0
        for parameter_combination_directory, method_parameter_combination_name, method_parameter_names, method_name in \
                methods_directory_paths:
            instance_sparsity_file_path = os.path.join(parameter_combination_directory, 'instance_sparsity.p')
            with open(instance_sparsity_file_path, mode='rb') as instance_sparsity_file:
                instance_sparsity_dictionary = pickle.load(instance_sparsity_file)
            assert isinstance(instance_sparsity_dictionary, dict)
            layer2_is = np.hstack(instance_sparsity_dictionary['layer2_percentage_of_active'])
            plt.hist(layer2_is, bins=10, density=False, color=colors[i], range=(0, 100))
            i += 1
            if arguments.instance_sparsity_separate_plots:
                plt.ylim([0, 5000000])
                plt.xlim([0, 100])
                plt.savefig(os.path.join(parameter_combination_directory, method_name + "_" + 'instance_sparsity' + '.png'))
                plt.close()

        if not arguments.instance_sparsity_separate_plots:
            plot_dictionary_path = os.path.join(os.getcwd(), "Plots", "instance_sparsity.png")
            # plt.ylim([0, 0.1])
            plt.ylim([0, 5000000])
            plt.xlim([0, 100])
            plt.savefig(plot_dictionary_path)
            plt.close()

    elif arguments.performance_summary_and_plot:
        i = 0

        average_training_performances = []
        me_training_performances = []
        ste_training_performances = []

        for parameter_combination_directory, method_parameter_combination_name, method_parameter_names, method_name in \
                methods_directory_paths:
            method_summary_file_path = os.path.join(parameter_combination_directory, 'method_summary.p')
            with open(method_summary_file_path, mode='rb') as method_summary_file:
                method_summary = pickle.load(method_summary_file)
            assert isinstance(method_summary, ParameterCombinationSummary)

            return_per_episode = method_summary.perf_meas
            r = lambda z: np.round(z, 2)
            avg_perf = method_summary.mean_perf
            stddev_perf = method_summary.stddev_perf
            sample_size = method_summary.sample_size
            uci, lci, me = compute_tdist_confidence_interval(avg_perf, stddev_perf, 0.05, sample_size)
            print("#------------------------- Method Name: " + method_name + "-------------------------#")
            print("The average return per episode is:", r(avg_perf))
            print("Standard deviation:", r(stddev_perf))
            print("Margin of error:", r(me))
            print("95% confidence interval: (" + str(r(uci)) + ", " + str(r(lci)) + ")")

            training_perf = method_summary.summaries['return_per_episode']
            avg_training_perf = np.average(training_perf, axis=0)
            stddev_training_perf = np.std(training_perf, axis=0, ddof=1)
            _, _, me_training_perf = compute_tdist_confidence_interval(avg_training_perf, stddev_training_perf, 0.05,
                                                                       method_summary.sample_size)
            average_training_performances.append(avg_training_perf)
            me_training_performances.append(me_training_perf)
            ste_training_performances.append(stddev_training_perf / np.sqrt(method_summary.sample_size))

        x = np.arange(NUMBER_OF_EPISODES) + 1
        for i in range(len(average_training_performances)):
            if arguments.performance_summary_and_plot_use_standard_error:
                plt.fill_between(x, ste_training_performances[i] - me_training_performances[i],
                                    ste_training_performances[i] + me_training_performances[i],
                                 color=lighter_colors[i])
            else:
                plt.fill_between(x, average_training_performances[i] - me_training_performances[i],
                                    average_training_performances[i] + me_training_performances[i],
                                 color=lighter_colors[i])
        for i in range(len(average_training_performances)):
            plt.plot(x, average_training_performances[i], color=colors[i])
        plt.xlim([0, 500])
        plt.ylim([-330, -120])
        plot_dictionary_path = os.path.join(os.getcwd(), "Plots", "avg_return_per_episode.png")
        plt.savefig(plot_dictionary_path, dpi=200)
        # plt.show()
        plt.close()

    elif arguments.buffer_size_results_plot:

        buffer_sizes = [100, 1000, 5000, 20000, 80000]

        methods_avg_performances = []
        methods_performances_me = []

        for method_name in arguments.method:
            print('#----------------------------- Method:', method_name, '-----------------------------#')
            temp_avg_performances = []
            temp_performances_me = []

            """ Method results directory """
            method_results_directory = os.path.join(env_dir, method_name)
            method_parameter_names = BEST_PARAMETERS_DICTIONARY[method_name]['ParameterNames']
            for size in buffer_sizes:
                """ Parameter combination directory """
                method_parameter_combination_name = 'BufferSize' + str(size)
                method_parameter_dictionary = BEST_PARAMETERS_DICTIONARY[method_name][size]
                for name in method_parameter_names[1:]:
                    method_parameter_combination_name += '_' + name + str(method_parameter_dictionary[name])
                parameter_combination_directory = os.path.join(method_results_directory,
                                                               method_parameter_combination_name)
                print("\n", parameter_combination_directory)
                if not os.path.exists(parameter_combination_directory):
                    raise ValueError("Couldn't find the directory for the method: " + method_name + "," + \
                                     "and the parameter combination: " + method_parameter_combination_name)
                else:
                    # If there is a directory for the given method and parameter combination, then store the parameters
                    # names and the path to the directory
                    method_summary_file_path = os.path.join(parameter_combination_directory, 'method_summary.p')
                    with open(method_summary_file_path, mode='rb') as method_summary_file:
                        method_summary = pickle.load(method_summary_file)
                    assert isinstance(method_summary, ParameterCombinationSummary)

                    method_summary.print_summary(2)

                    sample_size = method_summary.sample_size
                    avg_perf = method_summary.mean_perf
                    stddev_perf = method_summary.stddev_perf
                    _, _, me = compute_tdist_confidence_interval(avg_perf, stddev_perf, 0.05, sample_size)
                    temp_avg_performances.append(avg_perf)
                    temp_performances_me.append(me)

            print('#-----------------------------------------------------------------------------------#\n\n')
            methods_avg_performances.append(temp_avg_performances)
            methods_performances_me.append(temp_performances_me)

        x = np.arange(len(buffer_sizes), dtype=np.int64) + 1
        for i in range(len(methods_avg_performances)):
            temp_avg = np.array(methods_avg_performances[i], dtype=np.float64)
            temp_me = np.array(methods_performances_me[i], dtype=np.float64)

            plt.plot(x, temp_avg, color=colors[i])
            plt.errorbar(x, temp_avg, yerr=temp_me, color=colors[i])
        plt.ylim([-1100, -120])
        plt.xticks(ticks=x, labels=['100', '1000', '5000', '20000', '80000'])
        plot_directory_path = os.path.join(os.getcwd(), "Plots", "buffer_size_experiment.png")
        plt.savefig(plot_directory_path, dpi=200)
        plt.close()
