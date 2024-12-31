from Training import *
from ParamInitialization import*
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

#Function from course ML:
def evaluate_model(X, parameters, y, activation_fn):
    """
    Evaluate model with trained_parameters
    Parameters:
        X: Set of input to predict
        parameters: Dictionary of trained parameters
        y: true label
        activation_fn: Activation function
    """
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

def cross_validation(X_train, y_train, X_valid, y_valid, activation_function, lambda_values, nn_layers,
                     regularization_type):
    """
    Trained the network and find the best lambda to minimize validation loss
    Parameters:
        X_train: Training set
        y_train: true label associated with training set
        X_valid:  Validation set
        y_valid: true label associated with validation set
        activation_function: Activation function
        lambda_values: List of lambda values
        nn_layers: Neural network layers
        regularization_type: Type of regularization
        :return best lambda, value loss associated to the best lambda, trained parameters, cost list associated to the best lambda
    """


    best_lambda = None
    best_val_accuracy = 0
    best_parameters = None
    cost_per_lambda = {}  # Dictionary to save cost for each lambda
    stop_feedback = False  # Flag for terminal view

    def train_and_evaluate(lambd):
        # Train the model and return trained parameters and cost:

        trained_parameters, costs = train_model(
            X_train, y_train, nn_layers, activation_function, lambd, regularization_type
        )
        validation_accuracy, _, _, _ = evaluate_model(X_valid, trained_parameters, y_valid, activation_function)
        return lambd, validation_accuracy, trained_parameters, costs

    def feedback_message():
        for frame in itertools.cycle(['Training   ', 'Training.  ', 'Training.. ', 'Training...']):
            if stop_feedback:
                break
            print(f"\r{frame}", end='', flush=True)
            time.sleep(0.5)

    #Start the timer:
    start_time = time.time()

    # Thread for terminal view
    feedback_thread = threading.Thread(target=feedback_message)
    feedback_thread.start()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(train_and_evaluate, lambd): lambd for lambd in lambda_values}

        for future in futures:
            lambd = futures[future]
            try:
                result_lambda, val_accuracy, parameters, costs = future.result()
                print(f"\nLambda: {result_lambda}, Validation Accuracy: {val_accuracy}%")
                cost_per_lambda[result_lambda] = costs  # Save the cost for each lambda

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_lambda = result_lambda
                    best_parameters = parameters
            except Exception as e:
                print(f"\nErrore durante il calcolo per lambda={lambd}: {e}")

    # Stop terminal view
    stop_feedback = True
    feedback_thread.join()

    #Stop timer:
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTempo totale per il training: {total_time:.2f} secondi")

    return best_lambda, best_val_accuracy, best_parameters, cost_per_lambda
