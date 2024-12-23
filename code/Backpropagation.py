from ActivationFunction import*

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
        return A,Z
    elif activation_function == "tanh":
        A = tanh(Z)
        return A, Z
    elif activation_function == "sigmoid":
        A = sigmoid(Z)
        return A, Z
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
    store = {}

    A = X.T #At start, we use input data
    store["A0"] = A
    L = len(parameters) // 2 # Number of layers in the neural Networks
    for l in range(1, L+1):
        A_prev = A # The activation of the previous iteration becomes the current input
        Wl = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        if l < L:
            A, Z = linear_forward(A_prev, Wl, b, activation_function)
        else:
            A, Z = linear_forward(A, Wl, b, "sigmoid")
        #Store the cache for using during backprop:
        store[f"A{l}"] = A
        store[f"W{l}"] = parameters[f"W{l}"]
        store[f"Z{l}"] = Z
        store[f"A_prev{l}"] = A_prev
    return A, store

#Compute cost:
def compute_cost(A, Y, parameters, lambd, regularization = "L2"):
    m = Y.shape[1] # Number of samples
    b_cross_entropy =  -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

    #Add regularization:
    if regularization == "L2":
        L2_regularization = (lambd / (2 * m)) * sum(
            np.sum(np.square(parameters[f"W{l + 1}"])) for l in range(len(parameters) // 2))
        return b_cross_entropy + L2_regularization
    elif regularization == "L1":
        L1_regularization = (lambd / m) * sum(
            np.sum(np.abs(parameters[f"W{l + 1}"])) for l in range(len(parameters) // 2))
        return b_cross_entropy + L1_regularization
    else:
        #Else we don't use regularization
        return b_cross_entropy


def linear_backward(dA, cache, activation_function, lambd=0, regularization=None):
    """
    Calcola i gradienti per un singolo layer.
    """
    A_prev, W, Z = cache
    m = A_prev.shape[1]

    # Calcola dZ
    if activation_function == "sigmoid":
        dZ = dA * (sigmoid(Z) * (1 - sigmoid(Z)))
    elif activation_function == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    elif activation_function == "tanh":
        dZ = dA * (1 - np.tanh(Z) ** 2)
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")

    # Calcola dW, db e dA_prev
    dW = (1. / m) * np.dot(dZ, A_prev.T)
    if regularization == "L2":
        dW += (lambd / m) * W
    elif regularization == "L1":
        dW += (lambd / m) * np.sign(W)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def L_layer_backward(AL, Y, caches, parameters, lambd=0, regularization=None):
    """
    Esegue la retropropagazione per calcolare i gradienti dei pesi e dei bias.

    Arguments:
    AL -- attivazioni dell'output layer
    Y -- etichette vere
    caches -- dizionario contenente le attivazioni e i valori Z per ogni layer
    parameters -- dizionario contenente i pesi e i bias della rete
    lambd -- parametro di regolarizzazione
    regularization -- tipo di regolarizzazione ('L1' o 'L2')

    Returns:
    derivatives -- dizionario contenente i gradienti dei pesi e dei bias
    """

    m = Y.shape[1]  # numero di esempi
    L = len(parameters) // 2

    derivatives = {}

    # Calcola il gradiente dell'output layer
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Backward pass per l'output layer
    current_cache = (caches[f"A{L - 1}"], parameters[f"W{L}"], caches[f"Z{L}"])

    dA_prev, dW, db = linear_backward(dAL, current_cache, "sigmoid", lambd=lambd, regularization=regularization)

    # Salva i gradienti per l'output layer
    derivatives[f"dW{L}"] = dW
    derivatives[f"db{L}"] = db

    # Retropropagazione per i layer precedenti
    for l in reversed(range(1, L)):
        current_cache = (caches[f"A{l - 1}"], parameters[f"W{l}"], caches[f"Z{l}"])

        dA_prev, dW, db = linear_backward(dA_prev, current_cache, "sigmoid", lambd=lambd, regularization=regularization)

        # Salva i gradienti per il layer corrente
        derivatives[f"dW{l}"] = dW
        derivatives[f"db{l}"] = db

    return derivatives

