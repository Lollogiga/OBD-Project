from ParamInitialization import *
from Training import *

def cross_validate(X_train, y_train, X_valid, y_valid, activation_function, lambda_values, nn_layers, num_epochs=70,
                   learning_rate=0.01, batch_size=64):
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
        num_epochs: Numero di epoche per l'addestramento.
        learning_rate: Tasso di apprendimento.
        batch_size: Dimensione del mini-batch.

    Returns:
        best_lambda: Il valore di lambda che ha dato le migliori prestazioni sul validation set.
        best_val_accuracy: L'accuratezza corrispondente al miglior lambda.
        best_parameters: I parametri addestrati corrispondenti al miglior lambda.
    """

    best_lambda = None
    best_val_accuracy = 0
    best_cost = float('inf')
    best_parameters = None  # Per memorizzare i parametri migliori

    for lambd in lambda_values:
        # Inizializza i parametri
        parameters = param_init(activation_function, nn_layers)

        # Addestra il modello
        trained_parameters, costs = train_model(
            X_train, y_train, parameters, activation_function,
            lambd=lambd, num_epochs=num_epochs,
            learning_rate=learning_rate,
            regularization="L2",
            batch_size=batch_size,
            use_momentum=True
        )

        # Valuta sul validation set
        val_cost, val_accuracy = evaluate_model(X_valid, y_valid, trained_parameters, lambd,
                                                activation_function, regularization="L2")

        print(f"Lambda: {lambd}, Validation cost: {val_cost}, Validation accuracy: {val_accuracy * 100}%")

        # Controlla se questo è il miglior valore di lambda
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_lambda = lambd
            best_cost = val_cost
            best_parameters = trained_parameters  # Salva i parametri migliori

    return best_lambda, best_val_accuracy, best_parameters