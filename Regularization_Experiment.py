import numpy as np
import argparse
import os
import torch
import pickle

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Acrobot, MountainCar, PuddleWorld             # environments
from Experiment_Engine import Agent, RegularizedNeuralNetwork               # agent and function approximator

NUMBER_OF_EPISODES = 500


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.buffer_size = check_attribute_else_default(experiment_parameters, 'buffer_size', 20000)
        self.tnet_update_freq = check_attribute_else_default(experiment_parameters, 'tnet_update_freq', 10)
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'acrobot', 'puddle_world'])
        self.verbose = experiment_parameters.verbose
        # parameters specific to the parameter sweep
        self.learning_rate = check_attribute_else_default(exp_parameters, 'lr', 0.001)
        self.l1_reg = check_attribute_else_default(experiment_parameters, 'l1_reg', True)
        self.weights_reg = check_attribute_else_default(experiment_parameters, 'weights_reg', True)
        self.reg_factor = check_attribute_else_default(experiment_parameters, 'reg_factor', 0.1)

        environment_dictionary = {
            'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3},
            'acrobot': {'class': Acrobot, 'state_dims': 4, 'num_actions': 3},
            'puddle_world': {'class': PuddleWorld, 'state_dims': 2, 'num_actions': 4}
                                  }

        self.config = Config()
        self.config.store_summary = True
        self.summary = {}

        """ Parameters for the Environment """
            # Same for every experiment
        self.config.max_actions = 2000
        self.config.norm_state = True

        """ Parameters for the Function Approximator """
            # Same for every experiment
        self.config.state_dims = environment_dictionary[self.environment_name]['state_dims']
        self.config.num_actions = environment_dictionary[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.batch_size = 32
        self.config.training_step_count = 0
        self.config.gates = 'relu-relu'
            # Selected after finding the best parameter combinations for DQN with a given buffer size
        self.config.buffer_size = self.buffer_size
        self.config.tnet_update_freq = self.tnet_update_freq
            # These are the parameters that we are sweeping over
        self.config.lr = self.learning_rate
        self.config.reg_method = 'l1' if self.l1_reg else 'l2'
        self.config.weights_reg = self.weights_reg
        self.config.reg_factor = self.reg_factor

        self.env = environment_dictionary[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.fa = RegularizedNeuralNetwork(config=self.config, summary=self.summary)
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
    parser.add_argument('-env', action='store', default='acrobot', type=str,
                        choices=['mountain_car', 'acrobot', 'puddle_world'])
    parser.add_argument('-buffer_size', action='store', default=20000, type=np.int64)
    parser.add_argument('-tnet_update_freq', action='store', default=10, type=np.int64)
    parser.add_argument('-v', '--verbose', action='store_true')
    # parameters part of the parameter sweep
    parser.add_argument('-lr', action='store', default=0.001, type=np.float64,
                        choices=[0.01, 0.004, 0.001, 0.00025])
    parser.add_argument('-l1_reg', action='store_true')
    parser.add_argument('-weights_reg', action='store_true')
    parser.add_argument('-reg_factor', action='store', default=0.1, type=np.float64,
                        choices=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    exp_parameters = parser.parse_args()

    """ General results directory """
    results_parent_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(results_parent_directory):
        os.makedirs(results_parent_directory)
    """ Directory specific to the environment and the method """
    if exp_parameters.l1_reg:
        if exp_parameters.weights_reg:
            environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env,
                                                        'L1_Regularization_OnWeights')
        else:
            environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env,
                                                        'L1_Regularization_OnActivations')
    else:
        if exp_parameters.weights_reg:
            environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env,
                                                        'L2_Regularization_OnWeights')
        else:
            environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env,
                                                        'L2_Regularization_OnActivations')
    if not os.path.exists(environment_result_directory):
        os.makedirs(environment_result_directory)
    """ Directory specific to the parameters"""
    parameters_name = 'LearningRate' + str(exp_parameters.lr) \
                      + '_BufferSize' + str(exp_parameters.buffer_size) \
                      + '_Freq' + str(exp_parameters.tnet_update_freq) \
                      + "_RegFactor" + str(exp_parameters.reg_factor)
    parameters_result_directory = os.path.join(environment_result_directory, parameters_name)
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

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025}
# reg_factor = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
