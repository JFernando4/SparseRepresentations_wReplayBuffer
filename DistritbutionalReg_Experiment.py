import numpy as np
import argparse
import os
import torch
import pickle
import time

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Catcher3, MountainCar                         # environments
from Experiment_Engine import Agent, DistRegNeuralNetwork                   # agent and function approximator

ENVIRONMENT_DICTIONARY = {
    'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3, 'number_of_episodes': 500,
                     'saving_time': [50, 100, 250, 500], 'max_actions': 2000},
    'catcher': {'class': Catcher3, 'state_dims': 4, 'num_actions': 3, 'number_of_episodes': 1000000,
                'saving_time': [], 'max_actions': 500000},
}


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.learning_rate = check_attribute_else_default(exp_parameters, 'lr', 0.001)
        self.buffer_size = check_attribute_else_default(experiment_parameters, 'buffer_size', 20000)
        self.tnet_update_freq = check_attribute_else_default(experiment_parameters, 'tnet_update_freq', 10)
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'catcher'])
        self.verbose = experiment_parameters.verbose
        # parameters specific to distributional regularizers
        self.beta = check_attribute_else_default(experiment_parameters, 'beta', 0.1)
        self.reg_factor = check_attribute_else_default(experiment_parameters, 'reg_factor', 0.1)
        self.use_gamma = check_attribute_else_default(experiment_parameters, 'use_gamma', False)
        self.beta_lb = check_attribute_else_default(experiment_parameters, 'beta_lb', False)

        self.config = Config()
        self.config.store_summary = True
        self.summary = {}

        """ Parameters for the Environment """
        self.config.max_actions = ENVIRONMENT_DICTIONARY[self.environment_name]['max_actions']
        self.config.norm_state = True

        """ Parameters for the Function Approximator """
        self.config.state_dims = ENVIRONMENT_DICTIONARY[self.environment_name]['state_dims']
        self.config.num_actions = ENVIRONMENT_DICTIONARY[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.batch_size = 32
        self.config.training_step_count = 0

        self.config.lr = self.learning_rate
        self.config.buffer_size = self.buffer_size
        self.config.tnet_update_freq = self.tnet_update_freq
        self.config.gates = 'relu-relu'

        self.config.beta = self.beta
        self.config.reg_factor = self.reg_factor
        self.config.use_gamma = self.use_gamma
        self.config.beta_lb = self.beta_lb

        self.env = ENVIRONMENT_DICTIONARY[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.fa = DistRegNeuralNetwork(config=self.config, summary=self.summary)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        saving_times = ENVIRONMENT_DICTIONARY[self.environment_name]['saving_time']
        for i in range(ENVIRONMENT_DICTIONARY[self.environment_name]['number_of_episodes']):
            episode_number = i + 1
            self.rl_agent.train(1)
            if self.verbose and (((i+1) % 10 == 0) or i == 0):
                print("Episode Number:", episode_number)
                print('\tThe cumulative reward was:', self.summary['return_per_episode'][-1])
                print('\tThe cumulative loss was:', np.round(self.summary['cumulative_loss_per_episode'][-1], 2))
            if (episode_number in saving_times) and (self.environment_name != 'catcher'):
                self.save_network_params(suffix=str(episode_number) + 'episodes')
            if self.environment_name == 'catcher':
                assert isinstance(self.env, Catcher3)
                if self.env.timeout: break
        if self.environment_name == 'catcher':
            self.save_network_params(suffix='final')
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
    parser.add_argument('-run_number', action='store', default=1, type=int,
                        help='Number of the first run.')
    parser.add_argument('-env', action='store', default='mountain_car', type=str,
                        choices=['mountain_car', 'catcher'])
    parser.add_argument('-lr', action='store', default=0.001, type=np.float64)
    parser.add_argument('-buffer_size', action='store', default=20000, type=np.int64)
    parser.add_argument('-tnet_update_freq', action='store', default=10, type=np.int64)
    parser.add_argument('-v', '--verbose', action='store_true')
    # parameters specific to the method
    parser.add_argument('-beta', action='store', default=0.1, type=np.float64, choices=[0.1, 0.2, 0.5])
    parser.add_argument('-reg_factor', action='store', default=0.1, type=np.float64, choices=[0.1, 0.01, 0.001])
    parser.add_argument('-use_gamma', action='store_true')
    parser.add_argument('-layer2', action='store_true',
                        help='Indicates whether to apply regularization only to the second layer.')
    parser.add_argument('-beta_lb', action='store_true',
                        help='Indicates whether to enforce a lower bound on the value of beta.')
    parser.add_argument('-small_network', action='store_true')
    parser.add_argument('-runs', '---number_of_runs', action='store', default=1, type=int)
    exp_parameters = parser.parse_args()

    """ General results directory """
    results_parent_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(results_parent_directory):
        os.makedirs(results_parent_directory)

    """ Directory specific to the environment and the method """
    method_name = 'DistributionalRegularizers'
    if exp_parameters.use_gamma:
        method_name += '_Gamma'
    else:
        method_name += '_Beta'
    if exp_parameters.beta_lb:
        method_name += '_LowerBounded'
    environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env, method_name)
    if not os.path.exists(environment_result_directory):
        os.makedirs(environment_result_directory)

    """ Directory specific to the parameters"""
    parameters_name = 'LearningRate' + str(exp_parameters.lr) \
                      + '_BufferSize' + str(exp_parameters.buffer_size) \
                      + '_Freq' + str(exp_parameters.tnet_update_freq) \
                      + '_Beta' + str(exp_parameters.beta) \
                      + "_RegFactor" + str(exp_parameters.reg_factor)
    parameters_result_directory = os.path.join(environment_result_directory, parameters_name)
    if not os.path.exists(parameters_result_directory):
        os.makedirs(parameters_result_directory)

    for i in range(exp_parameters.number_of_runs):
        """ Directory specific to the run """
        agent_id = 'agent_' + str(exp_parameters.run_number + i)
        run_results_directory = os.path.join(parameters_result_directory, agent_id)
        print("The agent results directory is:", run_results_directory)
        if not os.path.exists(run_results_directory):
            os.makedirs(run_results_directory)

        """ Setting up and running the experiment """
        experiment = Experiment(experiment_parameters=exp_parameters, run_results_dir=run_results_directory)
        initial_time = time.time()
        experiment.run()
        final_time = time.time()
        print('Elapsed time in minutes:', (final_time - initial_time) / 60)

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025} for mountain car
# learning rate = {0.001, 0.00025, 0.0000625, 0.000015625} for catcher
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
