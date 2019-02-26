import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from Experiment_Engine import ParameterCombinationSummary, sample_activation_maps, compute_activation_map, \
    parse_method_parameters, get_method_results_directory, TwoLayerFullyConnected, compute_instance_sparsity

NUMBER_OF_EPISODES = 500

if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', action='store', default='mountain_car', type=str, choices=['mountain_car', 'acrobot',
                                                                                           'puddle_world'])
    parser.add_argument('-m', '--method', action='store', default='DQN', type=str,
                        choices=['DQN', 'DistributionalRegularizers_Gamma', 'DistributionalRegularizers_Beta'])
    parser.add_argument('-mp', '--method_parameters', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-sp', '--save_plots', action='store_true')
    arguments = parser.parse_args()

    parameters_dict = {
        'DQN': ['LearningRate', 'BufferSize', 'Freq'],
        'DistributionalRegularizers_Gamma': ['LearningRate', 'Beta', 'RegFactor'],
        'DistributionalRegularizers_Beta': ['LearningRate', 'Beta', 'RegFactor']
    }
    summary_names = ['return_per_episode', 'steps_per_episode', 'cumulative_loss_per_episode']
    perf_measure_name = 'return_per_episode'

    plots_dir = os.path.join(os.getcwd(), 'Plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    """ Method results """
    method_parameter_names = parameters_dict[arguments.method]
    method_parameter_combination = parse_method_parameters(parameter_names=method_parameter_names,
                                                           parameter_values=arguments.method_parameters)
    method_results_directory = get_method_results_directory(arguments.env, arguments.method)
    method_summary = ParameterCombinationSummary(
        param_comb_path=os.path.join(method_results_directory, method_parameter_combination),
        param_comb_name=method_parameter_combination, parameter_names=method_parameter_names,
        summary_names=summary_names, performance_measure_name=perf_measure_name
    )

    print("\n")
    print('#------------------------- Summary of method: -------------------------#')
    print("Method Name:", arguments.method)
    method_summary.print_summary(2)
    print('#----------------------------------------------------------------------#')
    print("\n")

    ###################
    #      Plots      #
    ###################
    runs = method_summary.runs
    first_idx = 0
    last_idx = len(runs) - 1
    median_idx = int(last_idx / 2)
    indices = [first_idx, median_idx, last_idx]

    print('The average return of the worst run was:', np.average(runs[first_idx]['summary']['return_per_episode']))
    print('The average return of the median run was:', np.average(runs[median_idx]['summary']['return_per_episode']))
    print('The average return of the best run was:', np.average(runs[last_idx]['summary']['return_per_episode']))

    colors = [
        '#FF8800',  # orange    -   worst
        '#FFCC00',  # yellow    -   median
        '#2A8FBD'  # blue      -   best
    ]

    """ Activation Maps """
    layer1_maps = []
    layer2_maps = []
    layers = [layer1_maps, layer2_maps]

    layer1_active_percentage = []
    layer2_active_percentage = []
    total_active_percentage = []
    all_active_percentages = [layer1_active_percentage, layer2_active_percentage, total_active_percentage]

    for idx in [first_idx, median_idx, last_idx]:
        network_weights_path = runs[idx]['weights_path']
        net = TwoLayerFullyConnected(input_dims=2, h1_dims=32, h2_dims=256, output_dims=3, gates='relu-relu')
        net.load_state_dict(torch.load(network_weights_path))
        net.eval()

        l1, l2 = compute_activation_map(net, 100)

        layer1_active, layer1_percentage = compute_instance_sparsity(l1)
        layer2_active, layer2_percentage = compute_instance_sparsity(l2)
        total_active = layer1_active + layer2_active
        total_percentage = (100 * total_active / (l1.shape[0] +l2.shape[0])).flatten()
        layer1_active_percentage.append(layer1_percentage)
        layer2_active_percentage.append(layer2_percentage)
        total_active_percentage.append(total_percentage)

        layer1_maps.extend(sample_activation_maps(l1, 5))
        layer2_maps.extend(sample_activation_maps(l2, 5))

    " Heat Maps"
    Nr = 3
    Nc = 5

    heatmap_names = ['_Layer1_Heatmap', '_Layer2_Heatmap']
    for k, layer_map in enumerate(layers):
        fig, axs = plt.subplots(Nr, Nc, figsize=(10, 10))
        images = []
        for i in range(Nr):
            for j in range(Nc):
                data = layer_map[i * Nc + j] / layer_map[i * Nc + j].max()
                images.append(axs[i, j].imshow(data))
                # axs[i, j].label_outer()
                axs[i, j].axis('off')

        # [left, bottom, width, height]
        cbaxes = fig.add_axes([0.075, 0.1, 0.85, 0.05])
        cb = plt.colorbar(images[0], cax=cbaxes, orientation='horizontal')
        # plt.tight_layout(pad=-10, w_pad=-10.0, h_pad=-35.0)
        plt.tight_layout(pad=-1, h_pad=-20)
        if arguments.save_plots:
            plt.savefig(os.path.join(plots_dir,
                                     arguments.method + "_" + method_parameter_combination + heatmap_names[k] + '.png'))
        else:
            plt.show()
        plt.close()

    " Instance sparsity plots "
    is_names = ['_Layer1_Instance_Sparsity', '_Layer2_Instance_Sparsity', '_Total_Instance_Sparsity']
    for i, activation_percentage in enumerate(all_active_percentages):
        for percentages, c in zip(activation_percentage, colors):
            plt.hist(percentages, bins=20, range=(0, 100), color=c)
            plt.ylim([0, 10000])

        if arguments.save_plots:
            plt.savefig(os.path.join(plots_dir,
                                     arguments.method + "_" + method_parameter_combination + is_names[i] + '.png'))
        else:
            plt.show()
        plt.close()
