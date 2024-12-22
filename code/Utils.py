import numpy as np
#Funzione dedicata al testing:
def check_parameter_shapes(parameters, nn_layers):
    """
    Verifica che le dimensioni dei parametri siano corrette.

    Parameters:
        parameters: dizionario contenente i pesi e bias
        nn_layers: lista contenente il numero di neuroni in ogni layer della rete
    """
    L = len(nn_layers)

    for l in range(1, L):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]

        # Verifica la forma di W[l]
        print(W.shape)
        assert W.shape == (nn_layers[l], nn_layers[l - 1]), f"Dimensione errata di W{l}: {W.shape}"

        # Verifica la forma di b[l]
        assert b.shape == (nn_layers[l], 1), f"Dimensione errata di b{l}: {b.shape}"

    print("Le dimensioni dei parametri sono corrette.")


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu_derivative(Z):
    return (Z > 0).astype(float)
