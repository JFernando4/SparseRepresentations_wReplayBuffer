import numpy as np
import argparse
import os
import torch
import pickle

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import MountainCar, Catcher3, PuddleWorld            # environments
from Experiment_Engine import Agent                                         # Agent
from Experiment_Engine import VanillaDQN, DistRegNeuralNetwork, \
    RegularizedNeuralNetwork, DropoutNeuralNetwork                          # agent and function approximator
from Experiment_Engine.util import BEST_PARAMETERS_DICTIONARY

ENVIRONMENT_DICTIONARY = {
    'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3, 'number_of_steps': 200000,
                     'max_episode_length': 200000},
    'catcher': {'class': Catcher3, 'state_dims': 4, 'num_actions': 3, 'number_of_steps': 500000,
                'max_episode_length': 500000},
    'puddle_world': {'class': PuddleWorld, 'state_dims': 2, 'num_actions': 4, 'number_of_steps': 200000,
                     'max_episode_length': 200000}
}


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.buffer_size = check_attribute_else_default(experiment_parameters, 'buffer_size', 20000)
        self.method = check_attribute_else_default(exp_parameters, 'method', 'DQN')
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'catcher', 'puddle_world'])
        parameters_dictionary = BEST_PARAMETERS_DICTIONARY[self.environment_name][self.method][self.buffer_size]
        self.verbose = experiment_parameters.verbose

        self.config = Config()
        self.config.store_summary = True
        # stored in summary: 'return_per_episode', 'loss_per_step', 'steps_per_episode', 'reward_per_step'
        self.summary = {}
        self.config.number_of_steps = ENVIRONMENT_DICTIONARY[self.environment_name]['number_of_steps']

        """ Parameters for the Environment """
        self.config.max_episode_length = ENVIRONMENT_DICTIONARY[self.environment_name]['max_episode_length']
        self.config.norm_state = True
        self.config.current_step = 0

        """ Parameters for the Function Approximator """
        self.config.state_dims = ENVIRONMENT_DICTIONARY[self.environment_name]['state_dims']
        self.config.num_actions = ENVIRONMENT_DICTIONARY[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.batch_size = 32

        # Parameters for any type of agent
        self.config.buffer_size = self.buffer_size
        self.config.lr = parameters_dictionary['LearningRate']
        self.config.tnet_update_freq = parameters_dictionary['Freq']

        if self.method in ['DRE', 'DRE_LB',
                           'DRG', 'DRG_LB']:
            self.config.beta = parameters_dictionary['Beta']
            self.config.reg_factor = parameters_dictionary['RegFactor']
            self.config.use_gamma = False
            self.config.beta_lb = False
            if self.method in ['DRG', 'DRG_LB']:
                self.config.use_gamma = True
            if self.method in ['DRE_LB', 'DRG_LB']:
                self.config.beta_lb = True
            self.fa = DistRegNeuralNetwork(config=self.config, summary=self.summary)

        elif self.method in ['L1A', 'L1W',
                             'L2A', 'L2W']:
            self.config.reg_factor = parameters_dictionary['RegFactor']
            self.config.reg_method = 'l1'
            if self.method in ['L2A', 'L2W']:
                self.config.reg_method = 'l2'
            self.config.weights_reg = False
            if self.method in ['L1W', 'L2W']:
                self.config.weights_reg = True
            self.fa = RegularizedNeuralNetwork(config=self.config, summary=self.summary)

        elif self.method in ['DQN']:
            self.fa = VanillaDQN(config=self.config, summary=self.summary)

        elif self.method in ['Dropout']:
            self.config.dropout_probability = parameters_dictionary['DropoutProbability']
            self.fa = DropoutNeuralNetwork(config=self.config, summary=self.summary)
        else:
            raise ValueError("No configuration available for the given method.")

        self.env = ENVIRONMENT_DICTIONARY[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        prev_idx = 0
        current_episode_number = 1
        assert hasattr(self.config, 'current_step')
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
    parser.add_argument('-buffer_size', action='store', default=20000, type=np.int64,
                        choices=[100, 1000, 5000, 20000, 80000])
    parser.add_argument('-m', '--method', action='store', type=str,
                        choices=['DQN',
                                 'DRE', 'DRE_LB',
                                 'DRG', 'DRG_LB',
                                 'L1A', 'L1W',
                                 'L2A', 'L2W',
                                 'Dropout'])
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()
    method = exp_parameters.method
    env_name = exp_parameters.env
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
    for name in BEST_PARAMETERS_DICTIONARY[env_name][method]['ParameterNames'][1:]:
        parameters_name += "_" + name + str(BEST_PARAMETERS_DICTIONARY[env_name][method][buffer_size][name])
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

