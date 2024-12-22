import numpy as np

from Utils import*


def sigmoid(a):
    return 1/(1+np.exp(-a))

def param_init(activation_function, nn_layers):
    """
    Inizializzazione dei pesi tramite:
        1) Xavier se l'utente seleziona funzione di attivazione tanh
        2) He se l'utente seleziona funzione di attivazione ReLU
    Parameters:
        activation_function: Funzione di attivazione selezionata dall'utente
        nn_layers: dimensione dei livelli nascosti
    Returns:
        parameters: dizionario contenente i pesi e i bias inizializzati
    """
    if activation_function == "relu":
        param = initialize_parameters_he(nn_layers)
    elif activation_function == "tanh":
        param = initialize_parameters_xavier(nn_layers)
    else:
        print("Error during param initialization")
        return -1
    check_parameter_shapes(param, nn_layers)
    return param

def initialize_parameters_xavier(nn_layers):
    """
    Inizializza i parametri usando Xavier initialization.

    Parameters:
        nn_layers: lista contenente il numero di neuroni in ogni layer della rete

    Returns:
        parameters: dizionario contenente i pesi e bias inizializzati
    """
    np.random.seed(1)  # per rendere i risultati riproducibili
    parameters = {}
    L = len(nn_layers)  # numero di layer nella rete

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(nn_layers[l], nn_layers[l - 1]) * np.sqrt(1 / nn_layers[l - 1])
        parameters[f"b{l}"] = np.zeros((nn_layers[l], 1))

        # Controllo delle dimensioni
        assert parameters[f"W{l}"].shape == (nn_layers[l], nn_layers[l - 1])
        assert parameters[f"b{l}"].shape == (nn_layers[l], 1)

    return parameters

def initialize_parameters_he(nn_layers):
    """
    Inizializza i parametri usando He initialization.

    Parameters:
        nn_layers: lista contenente il numero di neuroni in ogni layer della rete

    Returns:
        parameters: dizionario contenente i pesi e bias inizializzati
    """
    np.random.seed(1)  # per rendere i risultati riproducibili
    parameters = {}
    L = len(nn_layers)  # numero di layer nella rete

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(nn_layers[l], nn_layers[l - 1]) * np.sqrt(2 / nn_layers[l - 1])
        parameters[f"b{l}"] = np.zeros((nn_layers[l], 1))

        # Controllo delle dimensioni
        assert parameters[f"W{l}"].shape == (nn_layers[l], nn_layers[l - 1])
        assert parameters[f"b{l}"].shape == (nn_layers[l], 1)

    return parameters

#Implementazione backpropagation:
#Forward pass:
def linear_forward(A_prev, W, b, activation_function):
    """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A_prev -- activations from previous layer
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        A -- the output of the activation function, also called the post-activation value
        """
    Z = np.dot(W, A_prev) + b
    if activation_function == "relu":
        A = relu(Z)
        return A
    elif activation_function == "tanh":
        A = relu(Z)
        return A
    elif activation_function == "sigmoid":
        A = sigmoid(Z)
        return A
    else:
        print("Error during activation function")
        return -1

def L_layer_forward(X, parameters, activation_function):
    """
        Implement forward propagation for the model

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of param_init()

        Returns:
        A -- the output of the last layer
        A_prev -- the activations of the penultimate layer (useful for some applications)
        """
    A = X #At start, we use input data
    L = len(parameters) // 2 # Number of layers in the neural Networks
    for l in range(1, L+1):
        A_prev = A # The activation of the previous iteration becomes the current input
        Wl = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        if l < L:
            A = linear_forward(A_prev, Wl, b, activation_function)
        else:
            A = linear_forward(A, Wl, b, "sigmoid")
    return A, A_prev

