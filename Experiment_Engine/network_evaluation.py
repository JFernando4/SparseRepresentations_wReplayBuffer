import torch
import numpy as np

from .networks import TwoLayerFullyConnected


def compute_activation_map(network, granularity=100):
    """
    :param network: an instance of the class TwoLayerFullyConnected
    :param granularity: how fine should it be the partition on each direction
    :param sample_size: size of the sample
    :return: random sample of activation maps of non-dead neurons
    """
    assert isinstance(network, TwoLayerFullyConnected)
    assert network.fc1.in_features == 2                                     # The function admits only 2D state spaces

    layer1_num_neurons = 32
    layer2_num_neurons = 256
    partition_size = 2 / (granularity - 1)
    state_partition = np.arange(-1, 1 + partition_size, partition_size, dtype=np.float64)
    activation_maps_layer1 = np.zeros((layer1_num_neurons, granularity, granularity), dtype=np.float64)
    activation_maps_layer2 = np.zeros((layer2_num_neurons, granularity, granularity), dtype=np.float64)

    for i in range(granularity):
        for j in range(granularity):
            temp_state = np.array((state_partition[i], state_partition[j]), dtype=np.float64)
            x1, x2, _ = network.forward(temp_state, return_activations=True)
            activation_maps_layer1[:, i, j] = x1.detach().numpy()
            activation_maps_layer2[:, i, j] = x2.detach().numpy()

    return eliminate_dead_neuron_maps(activation_maps_layer1), eliminate_dead_neuron_maps(activation_maps_layer2)


def eliminate_dead_neuron_maps(activation_maps):
    """
    :param activation_maps: the activation maps of each neuron as computed by the function compute_activation_map.
                            Shape of activation_maps: (num_neurons, granularity, granularity)
                                num_neurons: number of neurons in the layer
                                granularity: number of partitions of each dimension of the state space
    :return: a numpy array of type np.float64 with dimensions (num_of_alive_neurons, granularity, granularity)
             which correspond to the activation maps of the alive neurons
    """
    assert isinstance(activation_maps, np.ndarray)
    assert len(activation_maps.shape) == 3
    indices = []
    number_of_neurons = activation_maps.shape[0]
    for i in range(number_of_neurons):
        total_activation = np.sum(activation_maps[i])
        if total_activation != 0:
            indices.append(i)
    indices = np.array(indices, dtype=np.int64)
    alive_neuron_maps = activation_maps[indices, :, :]
    return alive_neuron_maps


def sample_activation_maps(activation_maps, sample_size=10):
    alive_neuron_maps = eliminate_dead_neuron_maps(activation_maps)
    if sample_size > alive_neuron_maps.shape[0]:
        print("Not enough alive neurons for a sample size of " + str(sample_size) + ". Filling in with zeros.")
        # Fills in the missing maps with zeros
        for i in range(int(sample_size - alive_neuron_maps.shape[0])):
            alive_neuron_maps = np.row_stack((alive_neuron_maps, np.zeros(shape=(1,) + alive_neuron_maps.shape[1:],
                                                                          dtype=alive_neuron_maps.dtype)))
        return alive_neuron_maps
    else:
        sampled_indices = np.random.choice(alive_neuron_maps.shape[0], size=sample_size, replace=False)
        return alive_neuron_maps[sampled_indices, :, :]


def compute_instance_sparsity(activation_maps):
    assert isinstance(activation_maps, np.ndarray)
    if activation_maps.size == 0:   # all the neurons are dead
        active_neurons = 0
        percentage_active_neurons = 0.0
    else:
        sample_size = activation_maps.shape[0]
        positive_activations = np.int64((activation_maps > 0))
        active_neurons = np.sum(positive_activations, axis=0)
        percentage_active_neurons = ((active_neurons / sample_size) * 100).flatten()
    return active_neurons, percentage_active_neurons


def compute_activation_overlap(activation_maps, granularity=5):
    if activation_maps.size == 0:   # all the neurons are dead
        average_activation_overlap = 0.0
    else:
        am_shape = activation_maps[0].shape
        xincrement = int(am_shape[0] / (granularity-1))
        xpartition = np.arange(0, am_shape[0], xincrement, dtype=int)
        if (am_shape[0] % (granularity - 1)) == 0: xpartition = np.append(xpartition, am_shape[0] - 1)
        yincrement = int(am_shape[1] / (granularity-1))
        ypartition = np.arange(0, am_shape[1], yincrement, dtype=int)
        if (am_shape[1] % (granularity-1)) == 0: ypartition = np.append(ypartition, am_shape[1] - 1)

        average_activation_overlap = 0
        for act_map in activation_maps:
            downsampled_am = act_map[xpartition, :][:, ypartition]
            bool_am = (downsampled_am > 0).flatten()
            counter = 0
            map_activation_overlap = 0
            for i in range(len(bool_am) - 1):
                comparison_neurons = bool_am[(i+1):]
                counter += len(comparison_neurons)
                map_activation_overlap += np.sum(np.int64(np.logical_and(bool_am[i], comparison_neurons)))
            average_activation_overlap += (map_activation_overlap / counter)
    return average_activation_overlap
