import os
from dbm import error

import numpy as np
import pandas as pd


from DatasetPreprocessing import *
from NeuralNetwork import *


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
    """
    print(X_train.shape)
    nn_layers = [X_train.shape[1], 32, 32, 1]
    #Dobbiamo inizializzare i parametri:
    param = param_init(activation_function, nn_layers)
    train_model(X_train, y_train, param, lambdaL2_values[0], 10, 0.01, "L2")
if __name__ == "__main__":
    main()
