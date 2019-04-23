import numpy as np
# The pygame learning environment can be obtained from here:
#   https://github.com/ntasfi/PyGame-Learning-Environment
from ple.games.catcher import Catcher
from ple import PLE
import os
from Experiment_Engine.util import Config, check_attribute_else_default, check_dict_else_default


def get_ob(state):
    return np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])


def get_ob_normalize(state):
    np_state = np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])
    np_state[0] = (np_state[0] - 26) / 26
    np_state[1] = (np_state[1]) / 8
    np_state[2] = (np_state[2] - 26) / 26
    np_state[3] = (np_state[3] - 20) / 45
    return np_state


# 3-actions catcher
class Catcher3:
    """
    Environment Specifications:
    Short summary: Player control paddle and gains points for catching apples that fall from the sky; loses points and
                   lives otherwise.
    Number of Actions = 3 (move right, move left, do nothing)
    Observation Dimension = 4 (paddle x-position, paddle velocity, apple x-location, apple y-location)
    Observation Dtype = np.float64
    Reward =    1, if paddle touches apple
               -1, if apple touches floor and -1 life
               -5, if out of lives
                0, on any other transition
    Summary Name: steps_per_episode
    """

    def __init__(self, config, summary=None):
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             10000           The max number of actions executed before forcing
                                                                    a time out
        norm_state                  bool            True            Normalize the state to [-1,1]
        display                     bool            False           Whether to display the screen of the game
        init_lives                  int             3               Number of lives at the start of the game
        store_summary               bool            False           Whether to store the summary of the environment
        """
        self.max_actions = check_attribute_else_default(config, 'max_actions', default_value=10000)
        self.norm_state = check_attribute_else_default(config, 'norm_state', default_value=True)
        self.display = check_attribute_else_default(config, 'display', default_value=False)
        self.init_lives = check_attribute_else_default(config, 'init_lives', default_value=3)
        self.store_summary = check_attribute_else_default(config, 'store_summary', default_value=False)
        # setting up the summary dictionary
        self.summary = summary
        if self.store_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])
        # setting up original catcher environment with the specified parameters
        self.catcherOb = Catcher(init_lives=self.init_lives)
        if not self.display:
            # do not open a pygame window
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        if self.norm_state:
            self.pOb = PLE(self.catcherOb, fps=30, state_preprocessor=get_ob_normalize, display_screen=self.display)
        else:
            self.pOb = PLE(self.catcherOb, fps=30, state_preprocessor=get_ob, display_screen=self.display)
        self.pOb.init()
        # environment internal state
        self.actions = [100, 97, None]  # self.pOb.getActionSet() (right = 100, left = 97, do nothing = None)
        self.num_action = 3
        self.num_state = 4
        self.step_count = 0
        self.pOb.reset_game()
        self.current_state = self.pOb.getGameState()

    def _get_image(self):
        """return a np array with shape = [64, 64, 3]"""
        return self.pOb.getScreenGrayscale()

    def setseed(self, value):
        self.pOb.rng.seed(value)
        return 0

    def reset(self):
        if self.store_summary:
            self.summary["steps_per_episode"].append(self.step_count)
        self.pOb.reset_game()
        self.step_count = 0
        return self.pOb.getGameState()

    def step(self, a):
        reward = self.pOb.act(self.actions[a])
        self.step_count += 1
        terminate = self.pOb.game_over()
        self.current_state = self.pOb.getGameState()
        timeout = bool(self.step_count >= self.max_actions)
        return self.current_state, reward, terminate, timeout

    def get_current_state(self):
        return self.current_state

    def close(self):
        return


if __name__ == "__main__":
    config = Config()
    config.store_summary = True
    config.max_actions = 10000
    summary = {}
    actions = 3

    env = Catcher3(config, summary=summary)

    steps = 1000
    cumulative_reward = 0
    terminations = 0
    successful_episode_steps = []

    for i in range(steps):
        action = np.random.randint(actions)
        old_state = env.get_current_state()
        new_state, reward, terminate, timeout = env.step(action)
        print("Old state:",old_state, "-->",
              "Action:", action, "-->",
              "New state:", new_state)
        cumulative_reward += reward
        if terminate or timeout:
            print("\n## Reset ##\n")
            if terminate:
                terminations += 1
                successful_episode_steps.append(env.step_count)
            env.reset()

    print("Number of steps per episode:", summary['steps_per_episode'])
    print("Number of episodes that reached the end:", terminations)
    average_length = np.average(successful_episode_steps) if len(successful_episode_steps) > 0 else np.inf
    print("The average number of steps per episode was:", average_length)
    print("Cumulative reward:", cumulative_reward)
