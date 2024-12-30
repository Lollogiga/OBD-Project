import numpy as np
from Backpropragation import *
from ParamInitiliazizaion import *
from constant import *
from CrossValidation import *

def evaluate_model(X, parameters, y, activation_fn):

    probs, _ = L_layer_forward(X, parameters, activation_fn, "sigmoid")
    labels = (probs >= 0.5) * 1

    # accuracy
    accuracy = np.mean(labels == y) * 100

    # True Positives
    TP = np.sum((y == 1) & (labels == 1))
    # False Positives
    FP = np.sum((y == 0) & (labels == 1))
    # False Negatives
    FN = np.sum((y == 1) & (labels == 0))

    # precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # f1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision*100, recall*100, f1

def create_mini_batches(X, y, batch_size):
    """
    Splits the dataset into mini-batches of a given size.

    Parameters:
    X (numpy.ndarray): The input features of shape (num_samples, num_features).
    y (numpy.ndarray): The labels of shape (1, numSample).
    batch_size (int): The size of each mini-batch.

    Returns:
    list: A list of tuples, where each tuple contains a mini-batch (X_mini, y_mini).
    """
    # List to store mini-batches
    mini_batches = []
    y = y.T

    # Combine features (X) and labels (y) into a single array for shuffling
    data = np.hstack((X, y))

    # Shuffle the combined data to ensure randomness in mini-batches
    np.random.shuffle(data)

    # Calculate the number of complete mini-batches
    n_minibatches = data.shape[0] // batch_size

    # Loop through the data to create complete mini-batches
    for i in range(n_minibatches):
        # Extract a slice of data for the current mini-batch
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]

        # Separate features (X) from labels (y)
        X_mini = mini_batch[:, :-1]  # All columns except the last one
        Y_mini = mini_batch[:, -1].reshape((-1, 1))  # The last column as a column vector

        # Append the mini-batch as a tuple to the list
        mini_batches.append((X_mini, Y_mini))

    # Handle any remaining data that doesn't fit into a complete mini-batch
    if data.shape[0] % batch_size != 0:
        # Extract the remaining data
        mini_batch = data[n_minibatches * batch_size:data.shape[0], :]

        # Separate features (X) from labels (y)
        X_mini = mini_batch[:, :-1]  # All columns except the last one
        Y_mini = mini_batch[:, -1].reshape((-1, 1))  # The last column as a column vector

        # Append the last mini-batch to the list
        mini_batches.append((X_mini, Y_mini))

    # Return the list of mini-batches
    return mini_batches

def update_parameters(parameters, prev_parameters, grads, learning_rate, momentumBool, momentum):

    L = len(parameters)//2
    prev_parameters = parameters

    for l in range(1, L+1):
        if not momentumBool:
            parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"]
            parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"]
        else:
            parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"] + momentum * (parameters[f"W{l}"] - prev_parameters[f"W{l}"])
            parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"] + momentum * (parameters[f"b{l}"] - prev_parameters[f"b{l}"])

    return parameters, prev_parameters


def train_model(X_train, y_train, nn_layers, activation_function, lambd, regularitazion_type):
    """
        Addestra la rete neurale con mini-batch gradient descent o gradient descent con momentum.

        Args:
            X_train: Matrice di input (m, features).
            y_train: Matrice di etichette (1, m).
            nn_layers: Dimensione neural networks
            activation_function: type of activation function
            lambd: Parametro di regolarizzazione.
            num_epochs: Numero di epoche.
            Regularitazion_type: Tipo di regolarizzazione ('L2' o 'L1').
            learning_rate: Tasso di apprendimento.
            batch_size: Dimensione del mini-batch.
            momentum: Booleano che indica se usare il gradient descent o gradient descent with momentum.

        Returns:
            parameters: Parametri aggiornati.
            costs: Lista dei costi calcolati a ogni epoca.
        """

    cost = [] #Cost for each epoch
    learning_rate = LEARNING_RATE

    # Inizializza i parametri:
    parameters, prev_parameters = param_init(activation_function, nn_layers)

    for epoch in range(NUM_EPOCHS):
        #Create mini-batch:
        mini_batches = create_mini_batches(X_train, y_train, BATCH_SIZE)
        #Iteration on miniBatch:
        for X_batch, Y_batch in mini_batches:
            #TODO: Try to use Diminishing step-size

            #Forward pass:
            Al, caches = L_layer_forward(X_batch, parameters, activation_function)

            #Compute cost:
            cost = compute_cost(Al, Y_batch, parameters, lambd, regularitazion_type)

            #Backward propagation:
            grads = L_layer_backward(Al, Y_batch, caches, parameters, activation_function, lambd, regularitazion_type)

            #Update param:
            parameters, prev_parameters = update_parameters(parameters, prev_parameters, grads, learning_rate, MOMENTUM_BOOL, MOMENTUM)


    return parameters, cost


