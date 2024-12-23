from DatasetPreprocessing import *

from ParamInitialization import *
from Training import *


def print_menu(message, choice_number):
    """
        Stampa a schermo una lista di opzioni

        Parameters:
            message: Messaggio da stampare a schermo
            choice_number: Lista di scelte valide
    """
    while True:
        print(message)
        choice = input(f"Select a number between {choice_number}: ").strip().lower()
        if choice in choice_number:
            return choice
        else:
            print("Choice is not valid. Please select a valid input\n")


def main():
    dataset = print_menu(
        "Choice a dateset:\n" +
        "[1] Diabete_dataset\n"+
        "[2] Decidere scelta2\n"+
        "[3] Decidere scelta2\n",
        ["1","2","3"]
    )

    activation_function = print_menu(
        "What activation function do you want to use?\n"+
        "[1] relu\n"+
        "[2] tanh",
        ["1", "2"]
    )
    if activation_function == "1":
        activation_function = "relu"
    elif activation_function == "2":
        activation_function = "tanh"
    else:
        print("Error during define activation relu")
        return -1

    #Select dataset and preprocessing:
    if dataset == "1":
        dataset = pd.read_csv("../dataset/diabetes_dataset.csv")
        print('Dataset shape: %s' % (str(dataset.shape)))
        print("First 5 row:\n", dataset.head())
        X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess_data(dataset, "Diabetes_binary", 0.1, 0.2)
    elif dataset == "2":
        dataset = pd.read_csv("../dataset/")
        print('Dataset shape: %s' % (str(dataset.shape)))
        print("First 5 row:\n", dataset.head())
        X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess_data(dataset, "Diabetes_binary", 0.1, 0.2)
    elif dataset == "3":
        dataset = pd.read_csv("../dataset/")
        print('Dataset shape: %s' % (str(dataset.shape)))
        print("First 5 row:\n", dataset.head())
        X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess_data(dataset, "Diabetes_binary", 0.1, 0.2)
    else:
        print("Error during define dataset")
        return -1

    #TODO pulire precedenti test nelle cartelle di output
    #Strategia Cross-Validation:
    #Fissiamo una griglia di valori lambda:
    # Creiamo una lista di valori lambda da testare: #TODO Provare con altri valori
    lambdaL1_values = np.logspace(-6, 6, 13)  # da 10^-6 a 10^6
    lambdaL2_values = np.logspace(-6, 6, 13)  # da 10^-6 a 10^6

    """
    Definiamo la dimensione dei vari layer:
        Layer di input: ∈ R^n dove n è il numero di features
        Livelli nascosti ciascuno con un certo numero di neuroni
        Layer di output: abbiamo un'unica uscita 0/1
    """

    nn_layers = [X_train.shape[1] ,64, 32, 32, 1]
    #Dobbiamo inizializzare i parametri

    parameters = param_init(activation_function, nn_layers)
    # Chiamata alla funzione di training
    print("X_train shape: ", X_train.shape)
    trained_parameters, costs = train_model(
        X_train, y_train, parameters, lambdaL2_values[0], num_epochs=70, learning_rate=0.001,
        regularization="L2", batch_size=32, use_momentum=True
    )

    train_cost, train_accuracy = evaluate_model(X_train, y_train, trained_parameters, lambdaL2_values[0], regularization="L2")
    print(f"Training cost: {train_cost}, Training accuracy: {train_accuracy * 100}%")

    # Valutazione sul validation set
    val_cost, val_accuracy = evaluate_model(X_valid, y_valid, trained_parameters, lambdaL2_values[0], regularization="L2")
    print(f"Validation cost: {val_cost}, Validation accuracy: {val_accuracy * 100}%")

    # Valutazione sul test set
    test_cost, test_accuracy = evaluate_model(X_test, y_test, trained_parameters, lambdaL2_values[0], regularization="L2")
    print(f"Test cost: {test_cost}, Test accuracy: {test_accuracy * 100}%")

    #Proviamo a passare un'instanza di input:
    print(X_test.shape)
    print(y_test)
    #index = np.random.randint(0, X_test.shape[0])
    index = 1
    X_sample = X_test[index, :].reshape(1, -1)  # Aggiungi la dimensione del batch
    y_sample = y_test[:, index]

    # Forward pass sul campione
    Al_sample, cache = L_layer_forward(X_sample, trained_parameters, activation_function)

    # Converti la probabilità in una classe binaria (0 o 1)
    predicted_class = (Al_sample > 0.5).astype(int)  # Se Al_sample > 0.5, predici 1, altrimenti 0

    # Stampa il risultato
    print(f"Probabilità di classe positiva: {Al_sample}")
    print(f"Classe predetta: {predicted_class}")
    print(f"Etichetta reale: {y_sample}")


if __name__ == "__main__":
    main()
