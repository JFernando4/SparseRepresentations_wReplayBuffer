import numpy as np
import argparse
import os
import torch
import pickle
import time

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Catcher3, MountainCar                         # environments
from Experiment_Engine import Agent, RegularizedNeuralNetwork               # agent and function approximator

ENVIRONMENT_DICTIONARY = {
    'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3, 'number_of_steps': 200000,
                     'max_episode_length': 200000},
    'catcher': {'class': Catcher3, 'state_dims': 4, 'num_actions': 3, 'number_of_steps': 500000,
                'max_episode_length': 500000},
}


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.buffer_size = check_attribute_else_default(experiment_parameters, 'buffer_size', 20000)
        self.tnet_update_freq = check_attribute_else_default(experiment_parameters, 'tnet_update_freq', 10)
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'catcher'])
        self.verbose = experiment_parameters.verbose
        # parameters specific to the parameter sweep
        self.learning_rate = check_attribute_else_default(exp_parameters, 'lr', 0.001)
        self.l1_reg = check_attribute_else_default(experiment_parameters, 'l1_reg', True)
        self.weights_reg = check_attribute_else_default(experiment_parameters, 'weights_reg', True)
        self.reg_factor = check_attribute_else_default(experiment_parameters, 'reg_factor', 0.1)

        self.config = Config()
        self.config.store_summary = True
        # stored in summary: 'return_per_episode', 'loss_per_step', 'steps_per_episode', 'reward_per_step'
        self.summary = {}
        self.config.number_of_steps = ENVIRONMENT_DICTIONARY[self.environment_name]['number_of_steps']

        """ Parameters for the Environment """
            # Same for every experiment
        self.config.max_episode_length = ENVIRONMENT_DICTIONARY[self.environment_name]['max_episode_length']
        self.config.norm_state = True
        self.config.current_step = 0

        """ Parameters for the Function Approximator """
            # Same for every experiment
        self.config.state_dims = ENVIRONMENT_DICTIONARY[self.environment_name]['state_dims']
        self.config.num_actions = ENVIRONMENT_DICTIONARY[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.batch_size = 32
            # Selected after finding the best parameter combinations for DQN with a given buffer size
        self.config.buffer_size = self.buffer_size
        self.config.tnet_update_freq = self.tnet_update_freq
            # These are the parameters that we are sweeping over
        self.config.lr = self.learning_rate
        self.config.reg_method = 'l1' if self.l1_reg else 'l2'
        self.config.weights_reg = self.weights_reg
        self.config.reg_factor = self.reg_factor

        self.env = ENVIRONMENT_DICTIONARY[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.fa = RegularizedNeuralNetwork(config=self.config, summary=self.summary)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        prev_idx = 0
        current_episode_number = 1
        while self.config.current_step != self.config.number_of_steps:
            self.rl_agent.train(1)
            if self.verbose and ((current_episode_number % 10 == 0) or (current_episode_number - 1 == 0)):
                print("Episode Number:", current_episode_number)
                print('\tThe cumulative reward was:', self.summary['return_per_episode'][-1])
                print('\tThe cumulative loss was:',
                      np.round(np.sum(self.summary['loss_per_step'][prev_idx:]), 2))
                print('\tCurrent environment steps:', self.config.current_step)
                prev_idx = self.config.current_step
            current_episode_number += 1
        if self.verbose:
            print("Number of episodes completed:", len(self.summary['return_per_episode']))
            print("The total cumulative reward was:", np.sum(self.summary['reward_per_step']))
            print("Current environment steps:", self.config.current_step)
        self.save_network_params()
        self.save_run_summary()

    def save_network_params(self):
        params_path = os.path.join(self.run_results_dir, 'final_network_weights.pt')
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
                        choices=['mountain_car', 'catcher'])
    parser.add_argument('-buffer_size', action='store', default=20000, type=np.int64)
    parser.add_argument('-tnet_update_freq', action='store', default=10, type=np.int64)
    parser.add_argument('-v', '--verbose', action='store_true')
    # parameters part of the parameter sweep
    parser.add_argument('-lr', action='store', default=0.001, type=np.float64)
    parser.add_argument('-l1_reg', action='store_true')
    parser.add_argument('-weights_reg', action='store_true')
    parser.add_argument('-reg_factor', action='store', default=0.1, type=np.float64,
                        choices=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    exp_parameters = parser.parse_args()

    """ General results directory """
    results_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    """ Directory specific to the environment and the method """
    environment_directory = os.path.join(results_directory, exp_parameters.env)
    if exp_parameters.l1_reg:
        method_directory = os.path.join(environment_directory, 'L1')
    else:
        method_directory = os.path.join(environment_directory, 'L2')
    if exp_parameters.weights_reg:
        method_directory += 'W'
    else:
        method_directory += 'A'
    if not os.path.exists(method_directory):
        os.makedirs(method_directory)

    """ Directory specific to the parameters"""
    parameters_name = 'BufferSize' + str(exp_parameters.buffer_size) \
                      + '_Freq' + str(exp_parameters.tnet_update_freq)  \
                      + '_LearningRate' + str(exp_parameters.lr) \
                      + "_RegFactor" + str(exp_parameters.reg_factor)
    parameters_result_directory = os.path.join(method_directory, parameters_name)
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
    initial_time = time.time()
    experiment.run()
    final_time = time.time()
    print('Elapsed time in minutes:', (final_time - initial_time) / 60)

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025} for mountain car
# learning rate = {0.001, 0.00025, 0.0000625, 0.000015625} for catcher
# reg_factor = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
