import matplotlib.pyplot as plt
import os

def save_loss_plots(lossCost, dataset_name, activation_function, regularization_type):
    """
    Save loss graphs in a separate folder structure by dataset, activation function and regularization type.
    """
    # Create the output folder if it does not exist
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the folder for the dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create folder for activation function (ReLU or Tanh)
    activation_dir = os.path.join(dataset_dir, activation_function)
    if not os.path.exists(activation_dir):
        os.makedirs(activation_dir)

    # Create folder for regularization type (L1 or L2)
    regularization_dir = os.path.join(activation_dir, regularization_type)
    if not os.path.exists(regularization_dir):
        os.makedirs(regularization_dir)

    # Save graphs for each lambda value
    for lambd, loss in lossCost.items():
        # Create the subfolder for the lambda value
        lambd_dir = os.path.join(regularization_dir, f"lambda_{lambd}")
        if not os.path.exists(lambd_dir):
            os.makedirs(lambd_dir)

        # Crea loss graphs
        plt.plot(loss)
        plt.title(f"Loss for lambda = {lambd}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        # Save graph
        plot_filename = os.path.join(lambd_dir, f"loss_lambda_{lambd}.png")
        plt.savefig(plot_filename)
        plt.close()

    print(f"Graphs saved in {regularization_dir}")



def save_evaluation_results(accuracy, precision, recall, f1, lambd, dataset_name, activation_function,
                            regularization_type):
    """
    Save the evaluation results (accuracy, precision, recall, f1) to a file in the correct folder.
    The file will be saved with the name formed by the dataset, activation function, regularization type and lambda.
    """
    # Create the output folder if it does not exist
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the folder for the dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create folder for activation function (ReLU or Tanh)
    activation_dir = os.path.join(dataset_dir, activation_function)
    if not os.path.exists(activation_dir):
        os.makedirs(activation_dir)

    # Create folder for regularization type (L1 or L2)
    regularization_dir = os.path.join(activation_dir, regularization_type)
    if not os.path.exists(regularization_dir):
        os.makedirs(regularization_dir)

    # File name with the value of lambda
    file_name = f"lambda_{lambd}_evaluation.txt"
    file_path = os.path.join(regularization_dir, file_name)

    # Write result into the file
    with open(file_path, "w") as file:
        file.write(f"Lambda: {lambd}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")

    print(f"Evaluation results saved in {file_path}")