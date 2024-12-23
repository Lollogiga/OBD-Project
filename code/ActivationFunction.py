import numpy as np

#Activation Function:
def sigmoid(Z):

    A = 1 / (1 + np.exp(-Z))

    return A

def tanh(Z):

    A = np.tanh(Z)

    return A


def relu(Z):

    A = np.maximum(0, Z)

    return A

def tanh_derivative(dA, Z):

    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_derivative(dA, Z):

    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ

def sigmoid_derivative(dA, Z):
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ