from CrossValidation import *
from constant import *
from Backpropagation import *
from ParamInitialization import *

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
    """
    Update the parameters of the model with SDG with/without momentum.
    Parameters:
         parameters (dict): A dictionary containing all the parameters of the model.
         prev_parameters (dict): A dictionary containing all the previous parameters of the model.
         grads (dict): A dictionary containing all the gradients calculated with backpropagation.
         learning_rate (float): The learning rate of the model.
         momentumBool (bool): Whether to use momentum.
         momentum (float): The momentum factor.
    Return:
        parameters (dict): A dictionary containing all the new parameters of the model.
        prev_parameters (dict): A dictionary containing all the previous parameters of the model.
    """
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


def train_model(X_train, y_train, nn_layers, activation_function, lambd, regularization_type):
    """
    Train the model
    Parameters:
        X_train (numpy.ndarray): The input features to use for training
        y_train (numpy.ndarray): The true labels to use for loss calculation
        nn_layers (list): A list containing all the layers of the neural network
        activation_function (function): The activation function of the neural network
        lambd (float): The regularization parameter
        regularization_type (str): The type of regularization
    Returns:
          parameters (dict): A dictionary containing all the trained parameters of the model.
          epochs_cot (list): A list containing the cost for each epoch.
    """


    epoch_cost = [] #Cost for each epoch
    learning_rate = LEARNING_RATE
    decay_rate = 0.01

    #Parameter initialization:
    parameters, prev_parameters = param_init(activation_function, nn_layers)

    for epoch in range(NUM_EPOCHS):
        if DECAY_BOOL:
            learning_rate = LEARNING_RATE / (1 + decay_rate*epoch)

        #Create mini-batch:
        mini_batches = create_mini_batches(X_train, y_train, BATCH_SIZE)
        #Iteration on miniBatch:
        for X_batch, Y_batch in mini_batches:


            #Forward pass:
            Al, caches = L_layer_forward(X_batch, parameters, activation_function)

            #Compute cost:
            cost = compute_cost(Al, Y_batch, parameters, lambd, regularization_type)

            #Backward propagation:
            grads = L_layer_backward(Al, Y_batch, caches, parameters, activation_function, lambd, regularization_type)

            #Update param:
            parameters, prev_parameters = update_parameters(parameters, prev_parameters, grads, learning_rate, MOMENTUM_BOOL, MOMENTUM)

        #Compute the cost on training set for monitoring
        AL_epoch, store = L_layer_forward(X_train, parameters, activation_function)
        cost = compute_cost(AL_epoch, y_train.T, parameters, lambd, regularization_type)
        epoch_cost.append(cost)


    return parameters, epoch_cost
