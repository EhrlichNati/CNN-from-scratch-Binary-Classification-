import numpy as np
from utils import *
import h5py


def cat_data_loader():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def flatten_data(x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes):
    x_train = x_train_orig.reshape(-1, x_train_orig.shape[0])/255
    x_test = x_test_orig.reshape(-1, x_test_orig.shape[0])/255
    return x_train, y_train_orig, x_test, y_test_orig, classes



def net_init(layers):
    layers_parameters = {}
    L = len(layers)

    for layer_num in range(1, L):

        layers_parameters['weights_matrix ' + str(layer_num)] = np.random.randn(layers[layer_num], layers[layer_num-1])*np.sqrt(2/layers[layer_num-1])
        layers_parameters['bias_v ' + str(layer_num)] = np.zeros((layers[layer_num], 1))

    return layers_parameters



def forward_propagate(activation_prev, weights_matrix, bias_v, function):
    """Calculate linear output"""
    linear_output = np.dot(weights_matrix, activation_prev) + bias_v


    """Choose activation"""
    if function == "sigmoid":
        activation_output, activation_cache = sigmoid(linear_output)
    elif function == "relu":
        activation_output, activation_cache = relu(linear_output)

    linear_cache = activation_prev, weights_matrix, bias_v


    return activation_output, linear_cache, activation_cache




def L_forward_propagate(X, parameters):

    L = len(parameters)//2
    activation_output = X
    caches = []

    for layer in range(1, L):
        activation_prev = activation_output
        weights_matrix = parameters['weights_matrix ' + str(layer)]
        bias_v = parameters['bias_v ' + str(layer)]
        activation_output, linear_cache, activation_cache = forward_propagate(activation_prev, weights_matrix, bias_v, "relu")
        caches.append((activation_output, linear_cache, activation_cache))



    """Propagate to L(final) layer, with different activation function"""
    weights_matrix = parameters['weights_matrix ' + str(L)]
    bias_v = parameters['bias_v ' + str(L)]
    activation_output, linear_cache, activation_cache = forward_propagate(activation_output, weights_matrix, bias_v, "sigmoid")
    caches.append((activation_output, linear_cache, activation_cache))

    y_predict = activation_output
    return y_predict, caches




# TODO: write the math to devlop intuition.
def backward_propagate(linear_output_derivative, linear_cache):

    activation_prev, weights_matrix, bias_v = linear_cache
    m = activation_prev.shape[1]

    weights_derivative = (1/m)*np.dot(linear_output_derivative, activation_prev.T)
    bias_derivative = (1/m)*np.sum(linear_output_derivative, axis=1, keepdims=True)
    activation_prev_derivative = np.dot(weights_matrix.T, linear_output_derivative)

    return activation_prev_derivative, weights_derivative, bias_derivative



def backward_activation_init(activation_output_derivative, cache, function_type):

    activation_output, linear_cache, activation_cache = cache

    """ Calculate linear_output_derivative according to function_type """
    if function_type == 'sigmoid':
        linear_output_derivative = activation_output_derivative * sigmoid_derivative(activation_cache)
    elif function_type == 'relu':
        linear_output_derivative = activation_output_derivative * relu_derivative(activation_cache)

    activation_prev_derivative, weights_derivative, bias_derivative = backward_propagate(linear_output_derivative, linear_cache)

    return activation_prev_derivative, weights_derivative, bias_derivative



def L_backward_propagate(output_layer_activation, y_true, caches):
    # output_layer_activation = y_predict
    # L_linear_output_derivative = linear_output_derivative of the prediction.

    gradients = {}
    L = len(caches)


    # insert L-layer gradients before loop
    output_layer_activation_derivative = -np.divide(y_true, output_layer_activation) + np.divide(1 - y_true, 1 - output_layer_activation)
    curr_cache = caches[L-1]
    gradients['activation_derivative ' + str(L-1)], gradients['weights_derivative ' + str(L)], gradients['bias_derivative ' + str(L)] =\
        backward_activation_init(output_layer_activation_derivative, curr_cache, "sigmoid")


    for layer in reversed(range(L-1)):
        curr_cache = caches[layer]
        gradients['activation_derivative ' + str(layer)], gradients['weights_derivative ' + str(layer+1)], gradients['bias_derivative ' + str(layer+1)] = \
            backward_activation_init(gradients['activation_derivative ' + str(layer+1)], curr_cache, "relu")


    return gradients


def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2

    for layer in range(L):
        parameters["weights_matrix " + str(layer + 1)] = parameters["weights_matrix " + str(layer + 1)] - learning_rate * gradients["weights_derivative " + str(layer + 1)]
        parameters["bias_v " + str(layer + 1)] = parameters["bias_v " + str(layer + 1)] - learning_rate * gradients["bias_derivative " + str(layer + 1)]
    return parameters



def predict(X_test, y_test, parameters):
    y_predict, caches = L_forward_propagate(X_test, parameters)

    m = X_test.shape[1]
    predicted_labeled_vector = np.zeros((1, m))

    for index in range(m):
        if y_predict[0, index] > 0.5:
            predicted_labeled_vector[0, index] = 1
        else:
            predicted_labeled_vector[0, index] = 0

    return (1/m)*np.sum(np.where(predicted_labeled_vector == y_test, 1, 0))

