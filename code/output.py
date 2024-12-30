import os
import matplotlib.pyplot as plt


def save_loss_plots(lossCost, dataset_name, activation_function, regularization_type):
    """
    Salva i grafici delle loss in una struttura di cartelle separata
    per dataset, funzione di attivazione e tipo di regolarizzazione.
    """
    # Crea la cartella di output se non esiste
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crea la cartella per il dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Crea la cartella per la funzione di attivazione (ReLU o Tanh)
    activation_dir = os.path.join(dataset_dir, activation_function)
    if not os.path.exists(activation_dir):
        os.makedirs(activation_dir)

    # Crea la cartella per il tipo di regolarizzazione (L1 o L2)
    regularization_dir = os.path.join(activation_dir, regularization_type)
    if not os.path.exists(regularization_dir):
        os.makedirs(regularization_dir)

    # Salva i grafici per ciascun valore di lambda
    for lambd, loss in lossCost.items():
        # Crea la sottocartella per il valore di lambda
        lambd_dir = os.path.join(regularization_dir, f"lambda_{lambd}")
        if not os.path.exists(lambd_dir):
            os.makedirs(lambd_dir)

        # Crea il grafico della loss
        plt.plot(loss)
        plt.title(f"Loss for lambda = {lambd}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        # Salva il grafico nella sottocartella
        plot_filename = os.path.join(lambd_dir, f"loss_lambda_{lambd}.png")
        plt.savefig(plot_filename)
        plt.close()

    print(f"Graphs saved in {regularization_dir}")


import os


def save_evaluation_results(accuracy, precision, recall, f1, lambd, dataset_name, activation_function,
                            regularization_type):
    """
    Salva i risultati di valutazione (accuracy, precision, recall, f1) in un file nella cartella corretta.
    Il file sarà salvato con il nome formato dal dataset, funzione di attivazione, tipo di regolarizzazione e lambda.
    """
    # Crea la cartella di output se non esiste
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crea la cartella per il dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Crea la cartella per la funzione di attivazione (ReLU o Tanh)
    activation_dir = os.path.join(dataset_dir, activation_function)
    if not os.path.exists(activation_dir):
        os.makedirs(activation_dir)

    # Crea la cartella per il tipo di regolarizzazione (L1 o L2)
    regularization_dir = os.path.join(activation_dir, regularization_type)
    if not os.path.exists(regularization_dir):
        os.makedirs(regularization_dir)

    # Nome del file con il valore di lambda
    file_name = f"lambda_{lambd}_evaluation.txt"
    file_path = os.path.join(regularization_dir, file_name)

    # Scriviamo i risultati nel file
    with open(file_path, "w") as file:
        file.write(f"Lambda: {lambd}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")

    print(f"Evaluation results saved in {file_path}")