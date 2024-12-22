import numpy as np
def sigmoid(Z):
    """
    Compute the sigmoid of Z

    Arguments:
    Z -- A scalar or numpy array of any size.

    Return:
    A -- output of sigmoid(z), same shape as Z
    """
    A = 1 / (1 + np.exp(-Z))  # Sigmoid function formula
    return A

