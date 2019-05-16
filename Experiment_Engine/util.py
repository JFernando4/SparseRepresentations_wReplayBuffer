
BEST_PARAMETERS_DICTIONARY = {
    'mountain_car': {       # found by using a sweep with max sample size of 400
        'DQN': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.004},
            1000: {'Freq': 10, 'LearningRate': 0.004},
            5000: {'Freq': 10, 'LearningRate': 0.004},
            20000: {'Freq': 10, 'LearningRate': 0.001},
            80000: {'Freq': 10, 'LearningRate': 0.001},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate']
        },

        'DQN_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.01},
            1000: {'Freq': 10, 'LearningRate': 0.004},
            5000: {'Freq': 10, 'LearningRate': 0.004},
            20000: {'Freq': 10, 'LearningRate': 0.004},
            80000: {'Freq': 10, 'LearningRate': 0.004},
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

        'DistributionalRegularizers_Beta_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.001},
            1000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.1},
            5000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.01},
            20000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
            80000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.1, 'RegFactor': 0.01},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor']
        },

        'DistributionalRegularizers_Gamma': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.01},
            1000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
            5000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.1},
            20000: {'Freq': 10, 'LearningRate': 0.001, 'Beta': 0.5, 'RegFactor': 0.1},
            80000: {'Freq': 10, 'LearningRate': 0.001, 'Beta': 0.2, 'RegFactor': 0.1},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor']
        },

        'DistributionalRegularizers_Gamma_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.1},
            1000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.5, 'RegFactor': 0.1},
            5000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.01},
            20000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.2, 'RegFactor': 0.001},
            80000: {'Freq': 10, 'LearningRate': 0.004, 'Beta': 0.1, 'RegFactor': 0.001},
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

        'L1_Regularization_OnWeights_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0, 'RegFactor': 0},
            1000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            5000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            20000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            80000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
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

        'L1_Regularization_OnActivations_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.00025, 'RegFactor': 0.1},
            1000: {'Freq': 10, 'LearningRate': 0.004, 'RegFactor': 0.0005},
            5000: {'Freq': 10, 'LearningRate': 0.004, 'RegFactor': 0.001},
            20000: {'Freq': 10, 'LearningRate': 0.004, 'RegFactor': 0.0001},
            80000: {'Freq': 10, 'LearningRate': 0.001, 'RegFactor': 0.005},
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

        'L2_Regularization_OnWeights_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0, 'RegFactor': 0},
            1000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            5000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            20000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            80000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
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

        'L2_Regularization_OnActivations_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0, 'RegFactor': 0},
            1000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            5000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            20000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
            80000: {'Freq': 10, 'LearningRate': 0, 'RegFactor': 0},
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
        },

        'Dropout_SmallNetwork': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0, 'DropoutProbability': 0},
            1000: {'Freq': 10, 'LearningRate': 0, 'DropoutProbability': 0},
            5000: {'Freq': 10, 'LearningRate': 0, 'DropoutProbability': 0},
            20000: {'Freq': 10, 'LearningRate': 0, 'DropoutProbability': 0},
            80000: {'Freq': 10, 'LearningRate': 0, 'DropoutProbability': 0},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'DropoutProbability']
        }
    },

    'catcher': {    # found by using a sweep with max sample size of 102
        'DQN': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625},
            1000: {'Freq': 50, 'LearningRate': 0.0000625},
            5000: {'Freq': 200, 'LearningRate': 0.00025},
            20000: {'Freq': 200, 'LearningRate': 0.00025},
            80000: {'Freq': 400, 'LearningRate': 0.00025},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate']
        },

        'DistributionalRegularizers_Beta': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625, 'Beta': 0.5, 'RegFactor': 0.1},
            1000: {'Freq': 50, 'LearningRate': 0.0000625, 'Beta': 0.5, 'RegFactor': 0.1},
            5000: {'Freq': 200, 'LearningRate': 0.00025, 'Beta': 0.1, 'RegFactor': 0.001},
            20000: {'Freq': 200, 'LearningRate': 0.00025, 'Beta': 0.1, 'RegFactor': 0.01},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'Beta': 0.1, 'RegFactor': 0.1},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor']
        },

        'DistributionalRegularizers_Gamma': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625, 'Beta': 0.1, 'RegFactor': 0.01},
            1000: {'Freq': 50, 'LearningRate': 0.0000625, 'Beta': 0.2, 'RegFactor': 0.001},
            5000: {'Freq': 200, 'LearningRate': 0.00025, 'Beta': 0.5, 'RegFactor': 0.001},
            20000: {'Freq': 200, 'LearningRate': 0.00025, 'Beta': 0.1, 'RegFactor': 0.01},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'Beta': 0.1, 'RegFactor': 0.1},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'Beta', 'RegFactor']
        },

        'L1_Regularization_OnWeights': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            1000: {'Freq': 50, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            5000: {'Freq': 200, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            20000: {'Freq': 200, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
        },

        'L1_Regularization_OnActivations': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625, 'RegFactor': 0.0001},
            1000: {'Freq': 50, 'LearningRate': 0.0000625, 'RegFactor': 0.0001},
            5000: {'Freq': 200, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            20000: {'Freq': 200, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
        },

        'L2_Regularization_OnWeights': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625, 'RegFactor': 0.0001},
            1000: {'Freq': 50, 'LearningRate': 0.0000625, 'RegFactor': 0.0001},
            5000: {'Freq': 200, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            20000: {'Freq': 200, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
        },

        'L2_Regularization_OnActivations': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625, 'RegFactor': 0.0005},
            1000: {'Freq': 50, 'LearningRate': 0.0000625, 'RegFactor': 0.01},
            5000: {'Freq': 200, 'LearningRate': 0.0000625, 'RegFactor': 0.001},
            20000: {'Freq': 200, 'LearningRate': 0.0000625, 'RegFactor': 0.001},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'RegFactor': 0.0001},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'RegFactor']
        },

        'Dropout': {
            # Buffer Size
            100: {'Freq': 10, 'LearningRate': 0.0000625, 'DropoutProbability': 0.1},
            1000: {'Freq': 50, 'LearningRate': 0.0000625, 'DropoutProbability': 0.1},
            5000: {'Freq': 200, 'LearningRate': 0.00025, 'DropoutProbability': 0.1},
            20000: {'Freq': 200, 'LearningRate': 0.00025, 'DropoutProbability': 0.1},
            80000: {'Freq': 400, 'LearningRate': 0.00025, 'DropoutProbability': 0.1},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate', 'DropoutProbability']
        }
    }
}


class Config:
    """
    Used to store the arguments of all the functions in the package Experiment_Engine. If a function requires arguments,
    it's definition will be: func(config), where config has all the parameters needed for the function.
    """
    def __init__(self):
        pass


def check_attribute_else_default(object_type, attr_name, default_value, choices=None):
    if not hasattr(object_type, attr_name):
        print("Creating attribute", attr_name)
        setattr(object_type, attr_name, default_value)
    if choices:
        if getattr(object_type, attr_name) not in choices:
            raise ValueError("The possible values for this attribute are: " + str(choices))
    return getattr(object_type, attr_name)


def check_dict_else_default(dict_type, key_name, default_value):
    assert isinstance(dict_type, dict)
    if key_name not in dict_type.keys():
        dict_type[key_name] = default_value
    return dict_type[key_name]
