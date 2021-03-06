import numpy as np
import torch

from Experiment_Engine.networks import TwoLayerFullyConnected, TwoLayerDropoutFullyConnected, weight_init
from Experiment_Engine.util import *
from Experiment_Engine.Tilecoder3 import IHT, tiles


class NeuralNetworkFunctionApproximation:
    """
    Parent class for all the neural networks
    summary: loss_per_step
    """
    def __init__(self, config, summary=None):
        """
        Config --- class that contains all the parameters in used in an experiment.
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions             int             3                   Number of actions available to the agent
        gamma                   float           1.0                 discount factor
        epsilon                 float           0.1                 exploration parameter
        state_dims              int             2                   number of dimensions of the environment's states
        optim                   str             'sgd'               optimization method
        lr                      float           0.001               learning rate

        # DQN parameters
        batch_size              int             32                  minibatch size
        tnet_update_freq        int             10                  the update frequency of the target network

        # Parameters for storing summaries
        store_summary           bool            False               store the summary of the agent
        number_of_steps         int             500000              Total number of environment steps
        """
        assert isinstance(config, Config)
        check_attribute_else_default(config, 'current_step', 0)
        self.config = config

        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.epsilon = check_attribute_else_default(config, 'epsilon', 0.1)
        self.state_dims = check_attribute_else_default(config, 'state_dims', 2)
        self.optim = check_attribute_else_default(config, 'optim', 'sgd', choices=['sgd', 'adam', 'rmsprop'])
        self.lr = check_attribute_else_default(config, 'lr', 0.001)
        # DQN parameters
        self.batch_size = check_attribute_else_default(config, 'batch_size', 32)
        self.tnet_update_freq = check_attribute_else_default(config, 'tnet_update_freq', 10)
        self.replay_buffer = ReplayBuffer(config)
        # summary parameters
        self.store_summary = check_attribute_else_default(config, 'store_summary', False)
        self.number_of_steps = check_attribute_else_default(config, 'number_of_steps', 500000)
        if self.store_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            self.loss_per_step = np.zeros(self.number_of_steps, dtype=np.float64)
            check_dict_else_default(self.summary, 'loss_per_step', self.loss_per_step)

        self.h1_dims = 32       # fixed parameter
        self.h2_dims = 256      # fixed parameter

        # policy network
        self.net = TwoLayerFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                          output_dims=self.num_actions)
        self.net.apply(weight_init)
        # target network
        self.target_net = TwoLayerFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                                 output_dims=self.num_actions)
        self.target_net.apply(weight_init)

        if self.optim == 'sgd': self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.optim == 'adam': self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optim == 'rmsprop': self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

    def compute_return(self, reward, state, termination):
        # Computes the Qlearning return
        with torch.no_grad():
            av_function = torch.max(self.target_net.forward(state), dim=1)[0]
            next_step_bool = torch.from_numpy((1 - np.int64(termination))).float()
            qlearning_return = torch.from_numpy(reward).float() + next_step_bool * self.gamma * av_function
        return qlearning_return

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            with torch.no_grad():
                # it is extremely unlikely (prob = 0) for there to be two actions with exactly the same action value
                optim_action = self.net.forward(state).argmax().numpy()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)

    def save_summary(self, current_loss):
        if not self.store_summary:
            return
        self.summary['loss_per_step'][self.config.current_step - 1] = current_loss

    def update_target_network(self):
        if (self.config.current_step % self.tnet_update_freq) == 0:
            self.target_net.load_state_dict(self.net.state_dict())


class ReplayBuffer:

    def __init__(self, config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        state_dims              int             2                   number of dimensions of the environment's state
        buffer_size             int             100                 size of the buffer
        """
        self.state_dims = check_attribute_else_default(config, 'state_dims', 2)
        self.buffer_size = check_attribute_else_default(config, 'buffer_size', 100)

        """ inner state """
        self.start = 0
        self.length = 0

        self.state = np.empty((self.buffer_size, self.state_dims), dtype=np.float64)
        self.action = np.empty(self.buffer_size, dtype=int)
        self.reward = np.empty(self.buffer_size, dtype=np.float64)
        self.next_state = np.empty((self.buffer_size, self.state_dims), dtype=np.float64)
        self.next_action = np.empty(self.buffer_size, dtype=int)
        self.termination = np.empty(self.buffer_size, dtype=bool)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise KeyError()
        elif isinstance(idx, np.ndarray):
            if (idx < 0).any() or (idx >= self.length).any():
                raise KeyError()
        shifted_idx = self.start + idx
        s = self.state.take(shifted_idx, axis=0, mode='wrap')
        a = self.action.take(shifted_idx, axis=0, mode='wrap')
        r = self.reward.take(shifted_idx, axis=0, mode='wrap')
        next_s = self.next_state.take(shifted_idx, axis=0, mode='wrap')
        next_a = self.next_action.take(shifted_idx, axis=0, mode='wrap')
        terminate = self.termination.take(shifted_idx, axis=0, mode='wrap')
        return s, a, r, next_s, next_a, terminate

    def store_transition(self, transition):
        if self.length < self.buffer_size:
            self.length += 1
        elif self.length == self.buffer_size:
            self.start = (self.start + 1) % self.buffer_size
        else:
            raise RuntimeError()

        storing_idx = (self.start + self.length - 1) % self.buffer_size
        state, action, reward, next_state, next_action, termination = transition
        self.state[storing_idx] = state
        self.action[storing_idx] = action
        self.reward[storing_idx] = reward
        self.next_state[storing_idx] = next_state
        self.next_action[storing_idx] = next_action
        self.termination[storing_idx] = termination

    def sample(self, sample_size):
        if sample_size > self.length or sample_size > self.buffer_size:
            raise ValueError("The sample size is to large.")
        sampled_idx = np.random.randint(0, self.length, sample_size)                    # Sample any indices
        # sampled_idx = np.random.choice(self.length, size=sample_size, replace=False)  # Sample unique indices
        return self.__getitem__(sampled_idx)


class VanillaDQN(NeuralNetworkFunctionApproximation):

    def __init__(self, config, summary=None):
        super(VanillaDQN, self).__init__(config, summary)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            self.save_summary(0)
            return

        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        prediction = torch.squeeze(self.net(state).gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (qlearning_return - prediction).pow(2).mean()
        loss.backward()
        self.optimizer.step()

        self.save_summary(loss.detach().numpy())
        self.update_target_network()


class DistRegNeuralNetwork(NeuralNetworkFunctionApproximation):
    """
    Neural network with distributional regularizers. This is the implementation of the ReLu + SKL network from:
    "The Utility of Sparse Representations for Control in Reinforcement Learning"
        - Vincent Liu, Raksha Kumaraswamy, Lei Le, and Martha White
     """
    def __init__(self, config, summary=None):
        super(DistRegNeuralNetwork, self).__init__(config, summary=summary)
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        reg_factor              float           0.1                 
        beta                    float           0.1                 average max activation
        use_gamma               bool            False               whether to use a gamma distribution instead of beta
        layer2_reg              bool            False               whether to apply regularization only to the second
                                                                    layer          
        beta_lb                 float           False               whether to set a lower bound for beta                                 
        """
        self.config = config
        self.reg_factor = check_attribute_else_default(config, 'reg_factor', 0.1)
        self.beta = check_attribute_else_default(config, 'beta', 0.1)
        self.use_gamma = check_attribute_else_default(config, 'use_gamma', False)
        self.beta_lb = check_attribute_else_default(config, 'beta_lb', False)   # lower bounds beta by 0.1
        self.beta_fixed_lower_bound = 0.1

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            self.save_summary(0)
            return

        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        x1, x2, x3 = self.net.forward(state, return_activations=True)
        prediction = torch.squeeze(x3.gather(1, torch.from_numpy(action).view(-1,1)))
        # unregularized loss
        loss = (qlearning_return - prediction).pow(2).mean()
        if self.use_gamma:
            layer2_average = x2.mean()
            kld_layer2 = self.kld(layer2_average)
            loss += self.reg_factor * kld_layer2 * self.h2_dims
            if self.beta_lb:
                kld_lb_layer2 = self.kld_lb(layer2_average)
                loss += self.reg_factor * kld_lb_layer2 * self.h2_dims
        else:
            layer2_average = x2.mean(dim=0)
            kld_layer2 = self.kld(layer2_average)
            loss += self.reg_factor * kld_layer2
            if self.beta_lb:
                kld_lb_layer2 = self.kld_lb(layer2_average)
                loss += self.reg_factor * kld_lb_layer2
        loss.backward()
        self.optimizer.step()

        self.save_summary(loss.detach().numpy())
        self.update_target_network()

    def kld_derivative(self, beta_hats):
        # Note: you can use either kld_derivative or kld. Both results in the same gradient when using autograd.
        positive_beta_hats = beta_hats[beta_hats > self.beta]
        first_term = 1 / positive_beta_hats
        second_term = torch.pow(first_term, 2) * self.beta
        kld_derivative = torch.sum((first_term - second_term))
        return kld_derivative

    def kld(self, beta_hats):
        positive_beta_hats = beta_hats[beta_hats > 0]
        high_beta_hats = positive_beta_hats[positive_beta_hats > self.beta]
        # the original kl divergence is: log(beta_hat) + (beta / beta_hat) - log(beta) - 1
        # however, since beta doesn't depend on the parameters of the network, omitting the term -log(beta) - 1 doesn't
        # have any effect on the gradient.
        return torch.sum(torch.log(high_beta_hats) + (self.beta / high_beta_hats))

    def kld_lb(self, beta_hats):
        # this is the same as kld but for applied when beta is less than 0.05, which enforces a lower bound on beta
        positive_beta_hats = beta_hats[beta_hats > 0]
        low_beta_hats = positive_beta_hats[positive_beta_hats < self.beta_fixed_lower_bound]
        return torch.sum(torch.log(low_beta_hats) + (self.beta / low_beta_hats))


class RegularizedNeuralNetwork(NeuralNetworkFunctionApproximation):
    """
    Neural network with L1 or L2 regularization on the weights or the activations
    """
    def __init__(self, config, summary=None):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        reg_factor              float           0.1                 factor for the regularization method
        reg_method              string          'l1'                regularization method. Choices: 'none', 'l1', 'l2'
        weights_reg             bool            False               whether to apply regularization on the weights or
                                                                    the activations
        """
        super(RegularizedNeuralNetwork, self).__init__(config, summary=summary)
        self.reg_factor = check_attribute_else_default(config, 'reg_factor', 0.1)
        self.reg_method = check_attribute_else_default(config, 'reg_method', 'l1',
                                                       choices=['l1', 'l2'])
        self.weights_reg = check_attribute_else_default(config, 'weights_reg', False)

        if self.reg_method == 'l1':
            self.reg_function = torch.abs
        elif self.reg_method == 'l2':
            self.reg_function = lambda z: torch.pow(z, 2)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            self.save_summary(0)
            return

        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        x1, x2, x3 = self.net.forward(state, return_activations=True)
        # I don't like the line bellow because it is doing so many things. Here's a breakdown of what it does:
        # torch.squeeze - eliminates all the dimensions that are equal to 1
        # x3.gather - the first argument indicates the axis, the second argument indicates what item to gather
        # torch.from_numpy - converts the actions to a torch tensor
        # .view(-1, 1) - reshapes the tensor into tensor of shape batch_size x 1
        prediction = torch.squeeze(x3.gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (qlearning_return - prediction).pow(2).mean()
        reg_loss = 0
        if self.weights_reg:
            for name, param in self.net.named_parameters():
                # Regularization is only applied to the representation part of the network
                if not name.startswith('fc3'):
                    reg_loss += torch.sum(self.reg_function(param))
        else:
            reg_loss += torch.sum(self.reg_function(x2))
        loss += self.reg_factor * reg_loss
        loss.backward()
        self.optimizer.step()

        self.save_summary(loss.detach().numpy())
        self.update_target_network()


class DropoutNeuralNetwork(VanillaDQN):
    """
    The dropout neural network applies dropout only to the prediction from the policy network when computing the TD
    error of Q-learning. Otherwise, the models are set to eval() when computing the target of the TD error and when
    selecting actions --- which multiplies the activations by the dropout probability. Reasoning: Both the target and
    the actions should be computed using action-values that are as accurate as possible. We don't care that the neural
    network that computes them is sparse; we just care about the values being accurate.
    """
    def __init__(self, config, summary=None):
        assert isinstance(config, Config)
        super(DropoutNeuralNetwork, self).__init__(config, summary=summary)
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dropout_probability     float           0.5                 probability of setting activations to zero
        """
        self.dropout_probability = check_attribute_else_default(config, 'dropout_probability', 0.5)

        # policy network
        self.net = TwoLayerDropoutFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                                 output_dims=self.num_actions,
                                                 dropout_probability=self.dropout_probability)
        self.net.apply(weight_init)
        self.net.train()
        # target network
        self.target_net = TwoLayerDropoutFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                                        output_dims=self.num_actions,
                                                        dropout_probability=self.dropout_probability)
        self.target_net.apply(weight_init)
        self.target_net.eval()

        if self.optim == 'sgd': self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.optim == 'adam': self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optim == 'rmsprop': self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            self.net.eval()
            with torch.no_grad():
                optim_action = self.net.forward(state).argmax().numpy()
            self.net.train()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)


class TileCoderFA:

    def __init__(self, config=None):
        super().__init__()
        assert isinstance(config, Config)
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_tilings             int             32                  Number of tilings
        tiling_length           int             8                   The length of the tiling side
        num_actions             int             3                   Number of actions
        gamma                   float           1.0                 discount factor
        epsilon                 float           0.1                 exploration parameter
        state_dims              int             2                   Number of dimensions
        lr                      float           0.1                 Learning rate
        scaling_factor          np.array        [1,1]               The scaling factor and scaling offset are such so
        scaling_offset          np.array        [0,0]               (observation + scaling_offset) * scaling factor
                                                                    is within 0 and 1
        """
        self.num_tilings = check_attribute_else_default(config, 'num_tilings', 32)
        self.tiling_length = check_attribute_else_default(config, 'tiling_length', 8)
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.epsilon = check_attribute_else_default(config, 'epsilon', 0.1)
        self.state_dims = check_attribute_else_default(config, 'state_dims', 2)
        self.lr = check_attribute_else_default(config, 'lr', 0.1)
        self.scaling_factor = check_attribute_else_default(config, 'scaling_factor',
                                                           np.ones(self.state_dims, dtype=np.float64))
        self.scaling_offset = check_attribute_else_default(config, 'scaling_offset',
                                                           np.ones(self.state_dims, dtype=np.float64))

        # Why the self.tiling_length + 1? Because of the random off-set of each tiling, some tilings might not cover
        # the entire region of (self.tiling_length) ** self.state_dims. However, all the observations are scaled
        # down to that region. Hence, if we don't add the + 1, some observations might fall outside of the region
        # covered by the tilings.
        self.tiles_per_tiling = (self.tiling_length + 1) ** self.state_dims
        self.num_tiles = self.num_tilings * self.tiles_per_tiling
        self.theta = 0.001 * np.random.random(self.num_tiles * self.num_actions)
        self.iht = IHT(self.num_tiles)

    """ Scales input states to (0,1) and then multiplies by the side_length of a tiling """
    def scaling_function(self, state):
        assert len(state) == self.state_dims
        scaled_state = (state + self.scaling_offset) * self.scaling_factor * self.tiling_length
        return scaled_state

    """ Updates the value of the parameters corresponding to the state and action """
    def update(self, state, action, reward, next_state, next_action, termination):
        current_estimate = self.get_action_values(state)[action]
        qlearning_return = self.compute_return(reward, next_state, termination)
        value = qlearning_return - current_estimate
        scaled_state = self.scaling_function(state)

        tile_indices = np.asarray(
            tiles(self.iht, self.num_tilings, scaled_state), dtype=int) + (action * self.num_tiles)
        self.theta[tile_indices] += self.lr * value

    """ Returns the QLearning return of a specific state pair """
    def compute_return(self, reward, state, termination):
        max_av = np.max(self.get_action_values(state))
        qlearning_return = reward + (1 - np.int64(termination)) * max_av
        return qlearning_return

    def get_action_values(self, state):
        scaled_state = self.scaling_function(state)
        tile_indices = np.asanyarray(tiles(self.iht, self.num_tilings, scaled_state), dtype=np.int64)
        action_values = np.zeros(self.num_actions, dtype=np.float64)
        for i in range(self.num_actions):
            av = np.sum(self.theta[tile_indices + (i * self.num_tiles)])
            action_values[i] = av
        return action_values

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            argmax_av = np.int64(self.get_action_values(state).argmax())
            return argmax_av
        else:
            return np.random.randint(self.num_actions)

    def save_summary(self):
        pass
