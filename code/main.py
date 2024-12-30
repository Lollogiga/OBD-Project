import pandas as pd

from DatasetPreprocessing import *
from CrossValidation import *
from constant import *

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
    #Choice Dataset
    dataset = print_menu(
        "Choice a dateset:\n" +
        "[1] Wine Quality\n" +
        "[2] Mushroom dataset\n" +
        "[3] Heart attack risk dataset\n",
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
        "What regularitazion do you want to use?\n"+
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
    if dataset == "1":
        dataset = pd.read_csv("../dataset/WineQuality.csv")
        print('Dataset shape: %s' % (str(dataset.shape)))
        print("First 5 row:\n", dataset.head())
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasetPreprocessing(dataset, "quality_flag", 0.1, 0.2)
    elif dataset == "2":
        dataset = pd.read_csv("../dataset/mushroom_cleaned.csv")
        print("Dataset shape: %s" % (str(dataset.shape)))
        print("First 5 row\n", dataset.head(5))
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasetPreprocessing(dataset, "class", 0.1, 0.2)
    elif dataset == "3":
        dataset = pd.read_csv("../dataset/heart_disease_health.csv")
        print("Dataset shape: %s" % (str(dataset.shape)))
        print("First 5 row\n", dataset.head(5))
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasetPreprocessing(dataset, "HeartDiseaseorAttack", 0.1, 0.2)
    else:
        print("Invalid dataset")
        return -1

    #Set of lambda for cross-validation:
    #TODO: try with other values
    # Valori di L1 e L2 da testare
    lambdaL1_values = [1e-4, 1e-3, 0.01, 0.1, 0.3]
    lambdaL2_values = [1e-4, 1e-3, 0.01, 0.1, 0.3]

    if regularization_type=="L1":
        lambdaValues = lambdaL1_values
    elif regularization_type=="L2":
        lambdaValues = lambdaL2_values
    else:
        lambdaValues = -1
        print("Regularization_type not define")
        return lambdaValues

    """
        Definiamo la dimensione dei vari layer:
            Layer di input: ∈ R^n dove n è il numero di features
            Livelli nascosti ciascuno con un certo numero di neuroni
            Layer di output: abbiamo un'unica uscita 0/1
    """
    nn_layers = [X_train.shape[1], 32, 32, 1]


    cross_validation(X_train, y_train,
                     X_valid, y_valid,
                     activation_function,
                     lambdaValues,
                     nn_layers,
                     regularization_type,

    )


if __name__ == "__main__":
    main()
