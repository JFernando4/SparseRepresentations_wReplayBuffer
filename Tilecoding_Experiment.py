import numpy as np
import argparse
import os
import torch
import pickle
import time

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Catcher3, MountainCar                         # environments
from Experiment_Engine import Agent, TileCoderFA                            # agent and function approximator

ENVIRONMENT_DICTIONARY = {
    'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3, 'number_of_episodes': 500,
                     'saving_time': [50, 100, 250, 500], 'max_actions': 2000},
    'catcher': {'class': Catcher3, 'state_dims': 4, 'num_actions': 3, 'number_of_episodes': 1000000,
                'saving_time': [], 'max_actions': 500000},
}


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.num_tilings = check_attribute_else_default(experiment_parameters, 'num_tilings', 32)
        self.tiling_length = check_attribute_else_default(experiment_parameters, 'tiling_length', 10)
        self.learning_rate = check_attribute_else_default(exp_parameters, 'learning_rate', 0.001)
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'catcher'])
        self.verbose = experiment_parameters.verbose

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
        self.config.lr = self.learning_rate / self.num_tilings
        self.config.num_tilings = self.num_tilings
        self.config.tiling_length = self.tiling_length
        self.config.scaling_factor = 1/2
        self.config.scaling_offset = 1

        self.env = ENVIRONMENT_DICTIONARY[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.fa = TileCoderFA(config=self.config)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        for i in range(ENVIRONMENT_DICTIONARY[self.environment_name]['number_of_episodes']):
            episode_number = i + 1
            self.rl_agent.train(1)
            if self.verbose and (((i+1) % 10 == 0) or i == 0):
                print("Episode Number:", episode_number)
                print('\tThe cumulative reward was:', self.summary['return_per_episode'][-1])
            if self.environment_name == 'catcher':
                assert isinstance(self.env, Catcher3)
                if self.env.timeout: break
        self.save_run_summary()
        # self.save_tilecoder()

    def save_tilecoder(self):
        tilecoder_path = os.path.join(self.run_results_dir, 'tilecoder.p')
        with open(tilecoder_path, mode='wb') as tilecoder_file:
            pickle.dump(self.rl_agent.fa, tilecoder_file)

    def save_run_summary(self):
        total_reward = np.sum(self.summary['reward_per_step'])
        tr_path = os.path.join(self.run_results_dir, 'total_reward.p')
        with open(tr_path, mode='wb') as tr_file:
            pickle.dump(total_reward, tr_file)
        config_path = os.path.join(self.run_results_dir, 'config.p')
        with open(config_path, mode='wb') as config_file:
            pickle.dump(self.config, config_file)


if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_number', action='store', default=1, type=int)
    parser.add_argument('-env', action='store', default='mountain_car', type=str, choices=['mountain_car', 'catcher'])
    parser.add_argument('-nt', '--num_tilings', action='store', default=16, type=np.int64)
    parser.add_argument('-tl', '--tiling_length', action='store', default=8, type=np.int64)
    parser.add_argument('-lr', '--learning_rate', action='store', default=0.1, type=np.float64)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    """ General results directory """
    results_parent_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(results_parent_directory):
        os.makedirs(results_parent_directory)

    """ Directory specific to the environment and the method """
    environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env, 'TileCoding')
    if not os.path.exists(environment_result_directory):
        os.makedirs(environment_result_directory)

    """ Directory specific to the parameters"""
    parameters_name = 'LearningRate' + str(exp_parameters.learning_rate) \
                      + '_NumTilings' + str(exp_parameters.num_tilings) \
                      + "_TilingLength" + str(exp_parameters.tiling_length)
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
    initial_time = time.time()
    experiment.run()
    final_time = time.time()
    print('Elapsed time in minutes:', (final_time - initial_time)/60)

# Parameter Sweep:
# learning rate = {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1} for mountain car
# learning rate = {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1} for catcher
# num_tilings = {4, 8, 16, 32}
# tiling_length = {4, 8, 16, 32}
