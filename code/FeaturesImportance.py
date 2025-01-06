import os

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(feature_importance, feature_names, dataset_name, activaction_function, regularization_type):
    """
    Save plot about feature importance
    Parameters:
        feature_importance: Array of feature importance
        feature_names: Name of features.
        dataset_name: Name of dataset.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]

    sorted_indices = np.argsort(-feature_importance)
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_names, sorted_importance, color="teal")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()

    dir_path = "../output/" + dataset_name + "/" + activaction_function + "/" + regularization_type
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, f"{dataset_name}_features_importance.png")
    plt.savefig(file_path)


def compute_feature_importance(parameters):
    """
    Compute feature importance
    Parameters:
        parameters: Dictionary of parameters containing weight and bias
    Returns:
        feature_importance: Array of feature importance.
    """
    W1 = parameters["W1"]  # Weight of first layer
    feature_importance = np.sum(np.abs(W1), axis=0)  #Sum of absolute values along the nodes of the first layer
    return feature_importance
