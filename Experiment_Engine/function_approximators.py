import numpy as np
import torch

from Experiment_Engine.networks import TwoLayerFullyConnected, TwoLayerDropoutFullyConnected, weight_init
from Experiment_Engine.util import *


class NeuralNetworkFunctionApproximation:
    """ Parent class for all the neural networks """
    def __init__(self, config, summary=None):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions             int             3                   Number of actions available to the agent
        gamma                   float           1.0                 discount factor
        epsilon                 float           0.1                 exploration parameter
        state_dims              int             2                   number of dimensions of the environment's states
        optim                   str             'sgd'               optimization method
        lr                      float           0.001               learning rate
        gates                   str             relu-relu           types of gates for the network
        # DQN Parameters
        batch_size              int             32                  minibatch size
        training_step_count     int             0                   number of training steps so far
        tnet_update_freq        int             10                  the update frequency of the target network
        small_network           bool            False               if true, the network size is 32 x 32
        store_summary           bool            False               store the summary of the agent
                                                                    (cumulative_loss_per_episode)
        """
        assert isinstance(config, Config)
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.epsilon = check_attribute_else_default(config, 'epsilon', 0.1)
        self.state_dims = check_attribute_else_default(config, 'state_dims', 2)
        self.optim = check_attribute_else_default(config, 'optim', 'sgd', choices=['sgd', 'adam', 'rmsprop'])
        self.lr = check_attribute_else_default(config, 'lr', 0.001)

        self.gates = check_attribute_else_default(config, 'gates', 'relu-relu')
        self.batch_size = check_attribute_else_default(config, 'batch_size', 32)
        self.training_step_count = check_attribute_else_default(config, 'training_step_count', 0)
        self.tnet_update_freq = check_attribute_else_default(config, 'tnet_update_freq', 10)
        self.small_network = check_attribute_else_default(config, 'small_network', False)
        self.replay_buffer = ReplayBuffer(config)

        self.store_summary = check_attribute_else_default(config, 'store_summary', False)
        if self.store_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, 'cumulative_loss_per_episode', [])

        self.h1_dims = 32
        self.h2_dims = 256
        if self.small_network:
            self.h2_dims = 32
        print('Number of neurons in the first layer:', self.h1_dims)
        print('Number of neurons in the second layer:', self.h2_dims)

        self.cumulative_loss = 0
        # policy network
        self.net = TwoLayerFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                          output_dims=self.num_actions, gates=self.gates)
        self.net.apply(weight_init)
        # target network
        self.target_net = TwoLayerFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                                 output_dims=self.num_actions, gates=self.gates)
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
                optim_action = self.net.forward(state).argmax().numpy()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)

    def save_summary(self):
        if not self.store_summary:
            return
        self.summary['cumulative_loss_per_episode'].append(self.cumulative_loss)
        self.cumulative_loss = 0


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
            return

        self.training_step_count += 1
        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        prediction = torch.squeeze(self.net(state).gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (qlearning_return - prediction).pow(2).mean()
        loss.backward()
        self.optimizer.step()

        if self.store_summary:
            self.cumulative_loss += loss.detach().numpy()
        if (self.training_step_count % self.tnet_update_freq) == 0:
            self.target_net.load_state_dict(self.net.state_dict())


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
        self.layer2_reg = check_attribute_else_default(config, 'layer2_reg', False)
        self.beta_lb = check_attribute_else_default(config, 'beta_lb', False)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            return
        self.training_step_count += 1
        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        x1, x2, x3 = self.net.forward(state, return_activations=True)
        prediction = torch.squeeze(x3.gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (qlearning_return - prediction).pow(2).mean()
        if self.use_gamma:
            if not self.layer2_reg:
                layer1_average = x1.mean()
                kld_layer1 = self.kld(layer1_average)
                loss += self.reg_factor * kld_layer1 * self.h1_dims
                if self.beta_lb:
                    kld_lb_layer1 = self.kld_lb(layer1_average)
                    loss += self.reg_factor * kld_lb_layer1 * self.h1_dims
            layer2_average = x2.mean()
            kld_layer2 = self.kld(layer2_average)
            loss += self.reg_factor * kld_layer2 * self.h2_dims
            if self.beta_lb:
                kld_lb_layer2 = self.kld_lb(layer2_average)
                loss += self.reg_factor * kld_lb_layer2 * self.h2_dims
        else:
            if not self.layer2_reg:
                layer1_average = x1.mean(dim=0)
                kld_layer1 = self.kld(layer1_average)
                loss += self.reg_factor * kld_layer1
                if self.beta_lb:
                    kld_lb_layer1 = self.kld_lb(layer1_average)
                    loss += self.reg_factor * kld_lb_layer1
            layer2_average = x2.mean(dim=0)
            kld_layer2 = self.kld(layer2_average)
            loss += self.reg_factor * kld_layer2
            if self.beta_lb:
                kld_lb_layer2 = self.kld_lb(layer2_average)
                loss += self.reg_factor * kld_lb_layer2
        loss.backward()
        self.optimizer.step()
        if self.store_summary:
            self.cumulative_loss += loss.detach().numpy()
        if (self.training_step_count % self.tnet_update_freq) == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def kld_derivative(self, beta_hats):
        # Note: you can use either kld_derivative or kld. Both results in the same gradient.
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
        beta_fixed_lower_bound = 0.05
        low_beta_hats = positive_beta_hats[positive_beta_hats < beta_fixed_lower_bound]
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
            return

        self.training_step_count += 1
        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        x1, x2, x3 = self.net.forward(state, return_activations=True)
        # I don't like the line bellow because is doing so many things. Here's a breakdown of what it does:
        # torch.squeeze - eliminates all the dimensions that are equal to 1
        # x3.gather - the first argument indicates the axis, the second argument indicates what item to gather
        # torch.from_numpy - converts the actions to a torch tensor
        # .view(-1, 1) - reshapes the tensor into tensor of shape batch_size x 1
        prediction = torch.squeeze(x3.gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (qlearning_return - prediction).pow(2).mean()
        reg_loss = 0
        if self.weights_reg:
            for name, param in self.net.named_parameters():
                reg_loss += torch.sum(self.reg_function(param))
        else:
            reg_loss += torch.sum(self.reg_function(x1)) + torch.sum(self.reg_function(x2))
        loss += self.reg_factor * reg_loss
        loss.backward()
        self.optimizer.step()
        if self.store_summary:
            self.cumulative_loss += loss.detach().numpy()
        if (self.training_step_count % self.tnet_update_freq) == 0:
            self.target_net.load_state_dict(self.net.state_dict())


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
                                                 output_dims=self.num_actions, gates=self.gates,
                                                 dropout_probability=self.dropout_probability)
        self.net.apply(weight_init)
        self.net.train()
        # target network
        self.target_net = TwoLayerDropoutFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                                        output_dims=self.num_actions, gates=self.gates,
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
