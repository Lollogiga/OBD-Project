import numpy as np
from Utils import *

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
