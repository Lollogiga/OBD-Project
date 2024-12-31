from ActivationFunction import *

#Backpropagation
#Forward pass:
def linear_forward(A_prev, W, b, activation_function):
    """
    Implement the linear part of a layer's forward propagation.
    """

    # Compute the linear transformation Z
    Z = np.dot(W, A_prev) + b

    # Apply the activation function
    if activation_function == "relu":
        A = relu(Z)
    elif activation_function == "tanh":
        A = tanh(Z)
    elif activation_function == "sigmoid":
        A = sigmoid(Z)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")

    return A, Z


def L_layer_forward(X, parameters, activation_function, output_activation_function="sigmoid"):

    """
    Implement forward propagation for the model.

    Parameters:
        X_ data, numpy array of shape (m, n), where m = number of examples, n = number of features
        parameters: param of model
        activation_function: activation function to use for hidden layers
        output_activation_function: activation function for the output layer (default: sigmoid)

    Returns:
    A: the output of the last layer
    store: dictionary containing all intermediate values for backpropagation
    """
    store = {}

    A = X.T
    store["A0"] = A
    L = len(parameters) // 2  # Number of layers in the neural network

    for l in range(1, L + 1):
        A_prev = A  # The activation of the previous layer becomes the input for the current layer
        Wl = parameters[f"W{l}"]
        b = parameters[f"b{l}"]

        # Forward and activation:
        if l < L:
            A, Z = linear_forward(A_prev, Wl, b, activation_function)
        else:
            A, Z = linear_forward(A_prev, Wl, b, output_activation_function)

        # Store relevant values for backpropagation
        store[f"A{l}"] = A
        store[f"Z{l}"] = Z
        store[f"A_prev{l}"] = A_prev
    if A.shape != (1, X.shape[0]):
        print(f"Expected shape: {(1, X.shape[0])}, but got: {A.shape}")
        assert A.shape == (1, X.shape[0])
    return A, store


def compute_cost(A, Y, parameters, lambd, regularization):
    """
    Compute the cost
    Returns:
        float: The computed cost.
    """
    Y = Y.T
    m = Y.shape[1]  # Number of samples in the dataset


    # Clipping ensures that A remains in the range [1e-16, 1 - 1e-16].
    epsilon = 1e-16
    A = np.clip(A, epsilon, 1 - epsilon)

    # Verify shape of A and Y:
    assert A.shape == (1, m) and Y.shape == (1, m)

    # Calculate cross-entropy cost.
    # Formula: -1/m * sum(yi * log(Ai) + (1 - yi) * log(1 - Ai)) for i = 1,...,m
    b_cross_entropy = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Add regularization to the cost, if specified.
    if regularization == "L2":
        # Compute the L2 regularization term.
        # Formula: (lambda / (2 * m)) * sum(Wl^2) for all layers l
        L2_regularization = 0
        for l in range(len(parameters) // 2):
            W = parameters[f"W{l + 1}"]
            L2_regularization += np.sum(np.square(W))
        L2_regularization = (lambd / (2 * m)) * L2_regularization
        return b_cross_entropy + L2_regularization

    elif regularization == "L1":
        # Compute the L1 regularization term.
        # Formula: (lambda / m) * sum(|Wl|) for all layers l
        L1_regularization = 0
        for l in range(len(parameters) // 2):  # Loop through all layers
            W = parameters[f"W{l + 1}"]
            L1_regularization += np.sum(np.abs(W))
        L1_regularization = (lambd / m) * L1_regularization
        return b_cross_entropy + L1_regularization

    else:
        # If no regularization is specified, return only the cross-entropy cost.
        return b_cross_entropy


def linear_backward(dA, cache, activation_function, lambd=0, regularization=None):
    A_prev, W, Z = cache
    m = A_prev.shape[1]

    # Compute dZ using the function derivative
    if activation_function == "sigmoid":
        dZ = sigmoid_derivative(dA, Z)
    elif activation_function == "relu":
        dZ = relu_derivative(dA, Z)
    elif activation_function == "tanh":
        dZ = tanh_derivative(dA, Z)
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")

    #Compute dW, db and dA_prev
    dW = (1. / m) * np.dot(dZ, A_prev.T)

    # Apply regularization:
    if regularization == "L2":
        dW += (lambd / m) * W
    elif regularization == "L1":
        dW += (lambd / m) * np.sign(W)

    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def L_layer_backward(AL, Y, caches, parameters, activation_function="relu", lambd=0, regularization=None):
    """
    Implement the backward propagation for the model.
    Parameters:
        AL: the output of the last layer
        Y: the correct output of the last layer
        caches: dictionary containing all intermediate values for backpropagation
        parameters: param of model
        activation_function: activation function to use for hidden layers
        lambd: regularization parameter
        regularization: regularization parameter
    Returns:
        gradients: List of gradients
    """

    Y = Y.T
    m = Y.shape[1]  # Number of samples
    epsilon = 1e-12

    #AL \in [epsilon, 1-epsilon]
    AL = np.clip(AL, epsilon, 1 - epsilon)

    L = len(parameters) // 2  # Number of layer
    gradients = {}

    #Compute gradient of output layers:
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Backward pass for output layers:
    current_cache = (caches[f"A{L - 1}"], parameters[f"W{L}"], caches[f"Z{L}"])
    dA_prev, dW, db = linear_backward(dAL, current_cache, "sigmoid", lambd=lambd, regularization=regularization)

    #Save gradient:
    gradients[f"dW{L}"] = dW
    gradients[f"db{L}"] = db

    #Backpropagation for previous layers
    for l in reversed(range(1, L)):
        current_cache = (caches[f"A{l - 1}"], parameters[f"W{l}"], caches[f"Z{l}"])
        dA_prev, dW, db = linear_backward(dA_prev, current_cache, activation_function, lambd=lambd,
                                          regularization=regularization)

        #Save gradients:
        gradients[f"dW{l}"] = dW
        gradients[f"db{l}"] = db

    return gradients
