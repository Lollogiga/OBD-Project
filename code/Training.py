import numpy as np
from Backpropagation import *

def create_mini_batches(X, Y, batch_size):
    """
    Crea mini-batch da un set di dati.

    Args:
        X: Matrice di input.
        Y: Matrice di etichette
        batch_size: Dimensione dei mini-batch.

    Returns:
        mini_batches: Lista di tuple (X_batch, Y_batch).
    """
    m = X.shape[0]  # Numero totale di campioni
    mini_batches = []

    # Mix data
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation, :]  # (m, features)
    Y_shuffled = Y[:, permutation]  # (1, m)

    # Create mini-batch
    num_batches = m // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end, :]  # (batch_size, features)
        Y_batch = Y_shuffled[:, start:end]  # (1, batch_size)
        mini_batches.append((X_batch, Y_batch))

    # Manage the last batch (if m % batch != 0)
    if m % batch_size != 0:
        X_batch = X_shuffled[num_batches * batch_size:, :]
        Y_batch = Y_shuffled[:, num_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches


def update_parameters(parameters, grads, learning_rate, beta=0.9, velocity=None, use_momentum=False):
    """
    Funzione per aggiornare i parametri con o senza momentum.

    Args:
        parameters: Dizionario con i parametri della rete (pesi e bias).
        grads: Dizionario con i gradienti calcolati.
        learning_rate: Tasso di apprendimento.
        beta: Coefficiente di momentum (default=0.9).
        velocity: Dizionario con la velocità precedente (se si usa il momentum).
        use_momentum: Booleano che indica se usare il momentum o meno.

    Returns:
        parameters: Parametri aggiornati.
        velocity: Nuovo dizionario di velocità (usato solo se si usa il momentum).
    """
    if velocity is None:
        velocity = {f"W{l+1}": np.zeros_like(parameters[f"W{l+1}"]) for l in range(len(parameters) // 2)}
        velocity.update({f"b{l+1}": np.zeros_like(parameters[f"b{l+1}"]) for l in range(len(parameters) // 2)})

    for l in range(len(parameters) // 2):
        W_key = f"W{l+1}"
        b_key = f"b{l+1}"

        if use_momentum:
            # Aggiornamento con momentum
            velocity[W_key] = beta * velocity[W_key] + (1 - beta) * grads[f"dW{l+1}"]
            velocity[b_key] = beta * velocity[b_key] + (1 - beta) * grads[f"db{l+1}"]

            # Aggiornamento dei parametri
            parameters[W_key] -= learning_rate * velocity[W_key]
            parameters[b_key] -= learning_rate * velocity[b_key]
        else:
            # Aggiornamento senza momentum (gradient descent classico)
            parameters[W_key] -= learning_rate * grads[f"dW{l+1}"]
            parameters[b_key] -= learning_rate * grads[f"db{l+1}"]

    return parameters, velocity


def train_model(X, Y, parameters, lambd, num_epochs, learning_rate, regularization="L2", batch_size=32, use_momentum=False):
    """
    Addestra la rete neurale con mini-batch gradient descent o gradient descent con momentum.

    Args:
        X: Matrice di input (m, features).
        Y: Matrice di etichette (1, m).
        parameters: Dizionario con i parametri inizializzati (pesi e bias).
        lambd: Parametro di regolarizzazione.
        num_epochs: Numero di epoche.
        learning_rate: Tasso di apprendimento.
        regularization: Tipo di regolarizzazione ('L2' o 'L1').
        batch_size: Dimensione del mini-batch.
        use_momentum: Booleano che indica se usare il gradient descent o gradient descent with momentum.

    Returns:
        parameters: Parametri aggiornati.
        costs: Lista dei costi calcolati ad ogni epoca.
    """
    costs = []  # Per salvare il costo ad ogni epoca
    velocity = None  # Variabile per la velocità (momentum)

    for epoch in range(num_epochs):
        # Crea i mini-batch
        mini_batches = create_mini_batches(X, Y, batch_size)

        for X_batch, Y_batch in mini_batches:

            # Forward pass
            AL, caches = L_layer_forward(X_batch, parameters, "relu")

            # Calcola il costo
            cost = compute_cost(AL, Y_batch, parameters, lambd, regularization)

            # Backward pass
            grads = L_layer_backward(AL, Y_batch, caches, parameters, lambd, regularization)

            # Aggiorna i parametri (con o senza momentum)
            parameters, velocity = update_parameters(parameters, grads, learning_rate, use_momentum=use_momentum, velocity=velocity)

        # Calcola il costo sull'intero dataset (per monitoraggio)
        AL_epoch, _ = L_layer_forward(X, parameters, "relu")
        epoch_cost = compute_cost(AL_epoch, Y, parameters, lambd, regularization)
        costs.append(epoch_cost)

        # Stampa il costo ogni 10 epoche
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Cost after epoch {epoch}: {epoch_cost}")

    return parameters, costs
