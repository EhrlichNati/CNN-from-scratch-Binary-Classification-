import numpy as np
from utils import *
from model_functions import *



def train_model(x_train, y_true, x_test, y_test, layers, learning_rate, train_iterations, cost_function):

    # Initialize parameters
    parameters = net_init(layers)


    for iteration in range(train_iterations):

        # Forward propagate and extract caches
        y_predict, caches = L_forward_propagate(x_train, parameters)

        # Calculate cost
        cost_val = cost_function(y_predict, y_true)


        # Calculate gradients
        gradients = L_backward_propagate(y_predict, y_true, caches)

        # Descent
        parameters = update_parameters(parameters, gradients, learning_rate)



        if iteration % 100 == 0:
            print(iteration)
            #print("cost_val for " + f'{iteration}: ' + str(cost_val))

            pred_train = predict(x_train, y_true, parameters)
            print("pred_train", pred_train)

            pred_test = predict(x_test, y_test, parameters)
            print("pred_test", pred_test)

    return parameters







