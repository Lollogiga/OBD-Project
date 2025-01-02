import pandas as pd

from DatasetPreprocessing import *
from CrossValidation import *
from output import *
from FeaturesImportance import *

def print_menu(message, choice_number):
    """
        Print a list of choice

        Parameters:
            message: message to print
            choice_number: list of choice
    """
    while True:
        print(message)
        choice = input(f"Select a number between {choice_number}: ").strip().lower()
        if choice in choice_number:
            return choice
        else:
            print("Choice is not valid. Please select a valid input\n")

def main():
    #Choice Dataset
    dataset_choice = print_menu(
        "Choice a dataset:\n" +
        "[1] Wine Quality\n" +
        "[2] Mushroom dataset\n" +
        "[3] Airline passenger satisfaction\n",
    ["1", "2", "3"]
    )

    #Choice Activation function:
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
        print("Invalid activation function")
        return -1

    #Choice regularization:
    regularization_type = print_menu(
        "What regularization do you want to use?\n"+
        "[1] L1\n" +
        "[2] L2\n",
        ["1", "2"]
    )
    if regularization_type == "1":
        regularization_type = "L1"
    elif regularization_type == "2":
        regularization_type = "L2"
    else:
        print("Invalid regularization type")

    #Select Dataset and preprocessing Data:
    if dataset_choice == "1":
        dataset_name = "WineQuality"
        dataset = pd.read_csv("../dataset/WineQuality.csv")
        print('Dataset shape: %s' % (str(dataset.shape)))
        print("First 5 rows:\n", dataset.head())
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasetPreprocessing(dataset, "quality_flag", 0.1, 0.2)
        feature_names = dataset.columns.tolist()
    elif dataset_choice == "2":
        dataset_name = "Mushroom"
        dataset = pd.read_csv("../dataset/mushroom_cleaned.csv")
        print("Dataset shape: %s" % (str(dataset.shape)))
        print("First 5 rows:\n", dataset.head(5))
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasetPreprocessing(dataset, "class", 0.1, 0.2)
        feature_names = dataset.columns.tolist()
    elif dataset_choice == "3":
        dataset_name = "Airline"
        dataset = pd.read_csv("../dataset/transformed_airline_passenger_satisfaction.csv")
        print("Dataset shape: %s" % (str(dataset.shape)))
        print("First 5 rows:\n", dataset.head(5))
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasetPreprocessing(dataset, "satisfaction", 0.1, 0.2)
        feature_names = dataset.columns.tolist()
    else:
        print("Invalid dataset")
        return -1

    # Set of lambda for cross-validation:
    if regularization_type == "L1":
        lambdaValues = [1e-4, 1e-3, 0.01, 0.1, 0.3]
    elif regularization_type == "L2":
        lambdaValues = [1e-4, 1e-3, 0.01, 0.1, 0.3]
    else:
        lambdaValues = -1
        print("Regularization type not defined")
        return lambdaValues

    """
    Layer Dimension:
        -Input Layer: ∈ R^n (n number of features)
        -Hidden Layer: each with a specific number of neurons
        -Output Layers: We works with binary classification. have one output
    """

    nn_layers = [X_train.shape[1], 16, 32, 1]

    accuracyDictionary = {}

    #Start with Cross validation:
    lambd,_, parameters, lossCost = cross_validation(
        X_train, y_train, X_valid, y_valid,
        activation_function, lambdaValues,
        nn_layers, regularization_type
    )


    # Salva i grafici della loss
    save_loss_plots(lossCost, dataset_name, activation_function, regularization_type)

    # Valutazione sul test set
    accuracy, precision, recall, f1 = evaluate_model(X_test, parameters, y_test, activation_function)

    # Salviamo i risultati nel file
    save_evaluation_results(accuracy, precision, recall, f1, lambd, dataset_name, activation_function,
                            regularization_type)

    # Stampa dei risultati
    print(f"Lambda*: {lambd}")
    print(f"Accuracy on test set: {accuracy}")
    print(f"Precision on test set: {precision}")
    print(f"Recall on test set: {recall}")
    print(f"F1 Score on test set: {f1}\n")


    feature_importance = compute_feature_importance(parameters)
    plot_feature_importance(feature_importance, feature_names, dataset_name)

    W1 = parameters[f"W1"]
    df_W1 = pd.DataFrame(W1, columns=feature_names[:-1])
    dir_path = "../output/" + dataset_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, f"{dataset_name}_weight.csv")
    df_W1.to_csv(file_path, index=False)
if __name__ == "__main__":
    main()
