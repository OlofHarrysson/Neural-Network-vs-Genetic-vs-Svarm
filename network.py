import math
import numpy as np
import uuid
import random
import sys
from itertools import chain

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Neuron_layer():
    def __init__(self, nbr_neurons, nbr_inputs):
        epsi = 0.12 # Found value in Coursea machine learning course exersise 4
        self.weights = np.random.rand(nbr_neurons, nbr_inputs) *2*epsi-epsi # Matrix, nbr_neur rows, nbr_inputs cols
        self.bias = np.random.rand(nbr_neurons, 1) *2*epsi-epsi


class Neural_network():
    def __init__(self, nbr_input, nbr_hidden, nbr_output):
        self.nbr_input = nbr_input
        self.hidden_layer = Neuron_layer(nbr_hidden, nbr_input)
        self.output_layer = Neuron_layer(nbr_output, nbr_hidden)

    def SGD(self, X, Y, learn_rate):
        data_len = len(X)

        a1_list, z2_list, a2_list, a3_array, z3_list = self.feed_forward(data_len, X)

        error_L_array = a3_array - Y

        error_l2_list = []
        # Backwards propagation
        for i in range(data_len):
            temp = np.dot(self.output_layer.weights, error_L_array[-1])
            temp = temp[0]
            z2 = z2_list[-1]
            z2 = np.reshape(z2, (1, len(z2)))[0]
            error_l = temp * sigmoid_prime(z2)
            error_l2_list.append(error_l)

        # Delta1
        error_l2_list = np.transpose(error_l2_list)
        a1 = X
        delta1 = np.dot(error_l2_list, a1)

        # Delta2
        error_L_array = np.reshape(error_L_array, (data_len,1))
        delta2 = np.dot(np.transpose(error_L_array), a2_list)

        delta1_sum = np.sum(delta1, axis=0)
        delta2_sum = np.sum(delta2, axis=0)

        error_l2_sum = np.sum(error_l2_list, axis=1)
        error_L_sum = np.sum(error_L_array, axis=0)

        error_l2_sum = np.reshape(error_l2_sum, (len(error_l2_sum), 1))
        error_L_sum = np.reshape(error_L_sum, (len(error_L_sum), 1))

        # Update weights and bias
        self.hidden_layer.weights -= learn_rate / data_len * delta1_sum
        self.output_layer.weights -= learn_rate / data_len * delta2_sum

        self.hidden_layer.bias -= learn_rate / data_len * error_l2_sum
        self.output_layer.bias -= learn_rate / data_len * error_L_sum


    def feed_forward(self, data_len, X):
        a1_list = []
        z2_list = []
        a2_list = []
        a3_array = np.array([])
        z3_list = []

        for i in range(data_len):
            a1 = X[i]
            a1 = np.reshape(a1, (self.nbr_input, 1))

            # Hidden layer
            z2 = np.dot(self.hidden_layer.weights, a1) + self.hidden_layer.bias
            a2 = sigmoid(z2)
            a2 = list(chain(*a2)) # Flatten list

            # Output layer
            z3 = np.dot(self.output_layer.weights, a2) + self.output_layer.bias
            a3 = sigmoid(z3)

            z3 = list(chain(*z3)) # Flatten list

            a1_list.append(a1)
            z2_list.append(z2)
            a2_list.append(a2)
            a3_array = np.append(a3_array, a3)
            z3_list.append(z3)

        return a1_list, z2_list, a2_list, a3_array, z3_list

# TODO: Gradient checking
