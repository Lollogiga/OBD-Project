import numpy as np

def param_init(activation_function, nn_layers):
    """
    Parameters initialization with:
        1) Xavier for tanh activation function
        2) He for ReLU activation function
    Parameters:
        activation_function: Activation function
        nn_layers: Number of hidden layers
    Returns:
          Initialize parameters, and prev parameters (set to zero)
    """
    if activation_function == "relu":
        param = initialize_parameters_he(nn_layers)
    elif activation_function == "tanh":
        param = initialize_parameters_xavier(nn_layers)
    else:
        print("Error during param initialization")
        return -1, -1

    #Set prev_parameters
    prev_parameters = {key: np.zeros_like(value) for key, value in param.items()}

    return param, prev_parameters

def initialize_parameters_he(nn_layers):
    """
        Init parameters with He initialization
        Returns:
            parameters: dictionary of parameters
    """

    np.random.seed(3)
    parameters = {}
    L = len(nn_layers) - 1
    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(nn_layers[l], nn_layers[l - 1]) * np.sqrt(2.0 / nn_layers[l - 1])
        parameters['b' + str(l)] = np.zeros((nn_layers[l], 1))

    # Check shape
    assert parameters[f"W{l}"].shape == (nn_layers[l], nn_layers[l - 1])
    assert parameters[f"b{l}"].shape == (nn_layers[l], 1)

    return parameters

def initialize_parameters_xavier(nn_layers):
    """
       Init parameters with xavier initialization
       Returns:
           parameters: dictionary of parameters
    """
    np.random.seed(3)
    parameters = {}
    L = len(nn_layers)-1  # of layer

    for l in range(1, L+1):
        parameters[f"W{l}"] = np.random.randn(nn_layers[l], nn_layers[l - 1]) * np.sqrt(1 / nn_layers[l - 1])
        parameters[f"b{l}"] = np.zeros((nn_layers[l], 1))

        # Check shape
        assert parameters[f"W{l}"].shape == (nn_layers[l], nn_layers[l - 1])
        assert parameters[f"b{l}"].shape == (nn_layers[l], 1)

    return parameters
