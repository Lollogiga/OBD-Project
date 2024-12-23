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


#Activation Function:
def sigmoid(Z):

    A = 1 / (1 + np.exp(-Z))

    return A

def tanh(Z):

    A = np.tanh(Z)

    return A


def relu(Z):

    A = np.maximum(0, Z)

    return A

def tanh_derivative(dA, Z):

    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_derivative(dA, Z):

    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ

def sigmoid_derivative(dA, Z):
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ