from Training import *
from ParamInitiliazizaion import*
from concurrent.futures import ThreadPoolExecutor
import time

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


def cross_validation(X_train, y_train, X_valid, y_valid, activation_function, lambda_values, nn_layers,
                     regularitazion_type):
    best_lambda = None
    best_val_accuracy = 0
    best_parameters = None
    cost_per_lambda = {}  # Dizionario per salvare i costi per ogni lambda

    def train_and_evaluate(lambd):
        # Allena il modello e restituisce i costi
        trained_parameters, costs = train_model(
            X_train, y_train, nn_layers, activation_function, lambd, regularitazion_type
        )
        validation_accuracy, _, _, _ = evaluate_model(X_valid, trained_parameters, y_valid, activation_function)
        return lambd, validation_accuracy, trained_parameters, costs

    start_time = time.time()  # Inizia il timer

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(train_and_evaluate, lambd): lambd for lambd in lambda_values}

        for future in futures:
            lambd = futures[future]
            try:
                result_lambda, val_accuracy, parameters, costs = future.result()
                print(f"Lambda: {result_lambda}, Validation Accuracy: {val_accuracy}%")
                cost_per_lambda[result_lambda] = costs  # Salva i costi per lambda

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_lambda = result_lambda
                    best_parameters = parameters
            except Exception as e:
                print(f"Errore durante il calcolo per lambda={lambd}: {e}")

    end_time = time.time()  # Ferma il timer
    total_time = end_time - start_time
    print(f"Tempo totale per il training: {total_time:.2f} secondi")

    return best_lambda, best_val_accuracy, best_parameters, cost_per_lambda