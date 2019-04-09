import numpy as np
import argparse
import os
import torch
import pickle

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Acrobot, MountainCar, PuddleWorld             # environments
from Experiment_Engine import Agent                                         # Agent
from Experiment_Engine import VanillaDQN, DistRegNeuralNetwork, \
    RegularizedNeuralNetwork, DropoutNeuralNetwork                          # agent and function approximator

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
        100: {'Freq': 400, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},           # Wrong
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


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.buffer_size = check_attribute_else_default(experiment_parameters, 'buffer_size', 20000)
        self.method = check_attribute_else_default(exp_parameters, 'method', 'DQN')
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car'])
        self.verbose = experiment_parameters.verbose

        environment_dictionary = {
            'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3},
            'acrobot': {'class': Acrobot, 'state_dims': 4, 'num_actions': 3},
            'puddle_world': {'class': PuddleWorld, 'state_dims': 2, 'num_actions': 4}
                                  }

        self.config = Config()
        self.config.store_summary = True
        self.summary = {}

        """ Parameters for the Environment """
        self.config.max_actions = 2000
        self.config.norm_state = True

        """ Parameters for the Function Approximator """
        self.config.state_dims = environment_dictionary[self.environment_name]['state_dims']
        self.config.num_actions = environment_dictionary[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.batch_size = 32
        self.config.training_step_count = 0

        # Parameters for any type of agent
        self.config.buffer_size = self.buffer_size
        self.config.lr = BEST_PARAMETERS_DICTIONARY[self.method][self.buffer_size]['LearningRate']
        self.config.tnet_update_freq = BEST_PARAMETERS_DICTIONARY[self.method][self.buffer_size]['Freq']
        self.config.gates = 'relu-relu'

        if self.method in ['DistributionalRegularizers_Beta', 'DistributionalRegularizers_Gamma']:
            self.config.beta = BEST_PARAMETERS_DICTIONARY[self.method][self.buffer_size]['Beta']
            self.config.reg_factor = BEST_PARAMETERS_DICTIONARY[self.method][self.buffer_size]['RegFactor']
            self.config.use_gamma = False
            if self.method == 'DistributionalRegularizers_Gamma':
                self.config.use_gamma = True
            self.fa = DistRegNeuralNetwork(config=self.config, summary=self.summary)

        elif self.method in ['L1_Regularization_OnWeights', 'L1_Regularization_OnActivations',
                             'L2_Regularization_OnWeights', 'L2_Regularization_OnActivations']:
            self.config.reg_factor = BEST_PARAMETERS_DICTIONARY[self.method][self.buffer_size]['RegFactor']
            self.config.reg_method = 'l1'
            if self.method in ['L2_Regularization_OnWeights', 'L2_Regularization_OnActivations']:
                self.config.reg_method = 'l2'
            self.config.weights_reg = False
            if self.method in ['L1_Regularization_OnWeights', 'L2_Regularization_OnWeights']:
                self.config.weights_reg = True
            self.fa = RegularizedNeuralNetwork(config=self.config, summary=self.summary)

        elif self.method == 'DQN':
            self.fa = VanillaDQN(config=self.config, summary=self.summary)

        elif self.method == 'Dropout':
            self.config.dropout_probability = BEST_PARAMETERS_DICTIONARY[self.method][self.buffer_size]['DropoutProbability']
            self.fa = DropoutNeuralNetwork(config=self.config, summary=self.summary)
        else:
            raise ValueError("No configuration available for the given method.")

        self.env = environment_dictionary[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        saving_times = [50, 100, 250, 500]
        for i in range(NUMBER_OF_EPISODES):
            episode_number = i + 1
            self.rl_agent.train(1)
            if self.verbose and (((i+1) % 10 == 0) or i == 0):
                print("Episode Number:", episode_number)
                print('\tThe cumulative reward was:', self.summary['return_per_episode'][-1])
                print('\tThe cumulative loss was:', np.round(self.summary['cumulative_loss_per_episode'][-1], 2))
            if episode_number in saving_times:
                self.save_network_params(suffix=str(episode_number)+'episodes')
        self.save_run_summary()

    def save_network_params(self, suffix='50episodes'):
        params_path = os.path.join(self.run_results_dir, 'network_weights_' + suffix + '.pt')
        torch.save(self.fa.net.state_dict(), params_path)

    def save_run_summary(self):
        summary_path = os.path.join(self.run_results_dir, 'summary.p')
        with open(summary_path, mode='wb') as summary_file:
            pickle.dump(self.summary, summary_file)
        config_path = os.path.join(self.run_results_dir, 'config.p')
        with open(config_path, mode='wb') as config_file:
            pickle.dump(self.config, config_file)


if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_number', action='store', default=1, type=int)
    parser.add_argument('-env', action='store', default='mountain_car', type=str,
                        choices=['mountain_car'])
    parser.add_argument('-buffer_size', action='store', default=20000, type=np.int64,
                        choices=[100, 1000, 5000, 20000, 80000])
    parser.add_argument('-m', '--method', action='store', type=str,
                        choices=['DQN', 'DistributionalRegularizers_Beta', 'DistributionalRegularizers_Gamma',
                                 'L1_Regularization_OnWeights', 'L1_Regularization_OnActivations',
                                 'L2_Regularization_OnWeights', 'L2_Regularization_OnActivations',
                                 'Dropout'])
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()
    method = exp_parameters.method
    buffer_size = exp_parameters.buffer_size

    """ General results directory """
    results_parent_directory = os.path.join(os.getcwd(), 'Best_Parameters_Results')
    if not os.path.exists(results_parent_directory):
        os.makedirs(results_parent_directory)
    """ Directory specific to the environment """
    environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env)
    if not os.path.exists(environment_result_directory):
        os.makedirs(environment_result_directory)
    """ Method specific directory """
    method_result_directory = os.path.join(environment_result_directory, method)
    if not os.path.exists(method_result_directory):
        os.makedirs(method_result_directory)

    """ Directory specific to the parameters of the specific method and buffer size combination """
    parameters_name = 'BufferSize' + str(buffer_size)
    for name in BEST_PARAMETERS_DICTIONARY[method]['ParameterNames'][1:]:
        parameters_name += "_" + name + str(BEST_PARAMETERS_DICTIONARY[method][buffer_size][name])
    parameters_result_directory = os.path.join(method_result_directory, parameters_name)
    if not os.path.exists(parameters_result_directory):
        os.makedirs(parameters_result_directory)

    """ Directory specific to the run """
    agent_id = 'agent_' + str(exp_parameters.run_number)
    run_results_directory = os.path.join(parameters_result_directory, agent_id)
    print("The agent results directory is:", run_results_directory)
    if not os.path.exists(run_results_directory):
        os.makedirs(run_results_directory)

    """ Setting up and running the experiment """
    experiment = Experiment(experiment_parameters=exp_parameters, run_results_dir=run_results_directory)
    experiment.run()

