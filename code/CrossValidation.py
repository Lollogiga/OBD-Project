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
    """
    Esegue la cross-validation per trovare il miglior valore di lambda.

    Args:
        X_train: Dati di addestramento.
        y_train: Etichette di addestramento.
        X_valid: Dati di validazione.
        y_valid: Etichette di validazione.
        activation_function: Funzione di attivazione da utilizzare.
        lambda_values: Lista di valori di lambda da testare.
        nn_layers: Dimensione dei vari layer della rete.
        regularitazion_type: Tipo di regolarizzazione da utilizzare.

    Returns:
        best_lambda: Il valore di lambda che ha dato le migliori prestazioni sul validation set.
        best_val_accuracy: L'accuratezza corrispondente al miglior lambda.
        best_parameters: I parametri addestrati corrispondenti al miglior lambda.
    """
    best_lambda = None
    best_val_accuracy = 0
    best_parameters = None

    def train_and_evaluate(lambd):
        start_time = time.time()  # Avvia il timer
        # Allena il modello e valuta le prestazioni per un valore di lambda
        trained_parameters, cost = train_model(
            X_train, y_train, nn_layers, activation_function, lambd, regularitazion_type
        )
        validation_accuracy, _, _, _ = evaluate_model(X_valid, trained_parameters, y_valid, activation_function)
        end_time = time.time()  # Fine del timer
        elapsed_time = end_time - start_time  # Calcola il tempo trascorso
        return lambd, validation_accuracy, trained_parameters, elapsed_time

    # Usa ThreadPoolExecutor per eseguire i calcoli in parallelo
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(train_and_evaluate, lambd): lambd for lambd in lambda_values}

        for future in futures:
            lambd = futures[future]
            try:
                result_lambda, val_accuracy, parameters, elapsed_time = future.result()
                print(
                    f"Lambda: {result_lambda}, Validation Accuracy: {val_accuracy:.2f}, Time: {elapsed_time:.2f} seconds")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_lambda = result_lambda
                    best_parameters = parameters
            except Exception as e:
                print(f"Errore durante il calcolo per lambda={lambd}: {e}")

    return best_lambda, best_val_accuracy, best_parameters