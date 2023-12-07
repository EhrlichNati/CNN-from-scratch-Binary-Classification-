import numpy as np


def sigmoid(x):
    # x= linear_output (Z)
    return 1 /(1 + np.exp(-x)), x

def sigmoid_derivative(x):
    s = sigmoid(x)[0]
    return s * (1 - s)

##########################################

def relu(x):
    # x= linear_output (Z)
    return np.maximum(0, x), x

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

##########################################

""" Cost for regression  """
def cross_entropy_cost(y_predict, y_true):
    # y_predict = activation_output in l layer, np.arr shape (l.size, 1)
    # y_true = labels, np.arr shape (l.size, 1)
    # m = num of train example in X (input data)

    m = y_true.shape[1]

    return -(1./m)*np.sum(y_true*np.log(y_predict) + (1 - y_true)*np.log(1 - y_predict))


