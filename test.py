import numpy as np
import matplotlib
from enum import Enum

def id(x):
    return x

def Sigmoid(a): 
    def sigmoid(x):
        return 1 / (1 + np.exp((a * -x)))
    return sigmoid

def Tanh(a):
    def tanh(x):
        return np.sum((2 * Sigmoid(a)(x)), -1)
    return tanh

class Type(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:

    def __init__(self, neurons, weights, activation_fun, layer_type):
        self.neurons = neurons
        self.type = layer_type
        self.activation = activation_fun
        if layer_type == Type.INPUT:
            self.weights = weights
            self.matrix = np.eye(neurons)
        else:
            self.weights = weights + 1
            self.matrix = np.random.random((self.neurons, self.weights))

    def net(self, o):
        if self.type != Type.INPUT:
            o = np.concatenate((np.array([1]), o))
        return np.dot(self.matrix, o)

    
    def act(self, o):
        f = np.vectorize(self.activation)
        return f(self.net(o))

class Network:

    def __init__(self, hidden_layers_number, input_dimension, layer_length, act_arr):
        self.depth = hidden_layers_number
        self.input_layer = Layer(input_dimension, input_dimension, id, Type.INPUT)
        self.hidden_layers = np.empty(self.depth, dtype=object)
        self.hidden_layers[0] = Layer(layer_length[0], input_dimension, act_arr[0], Type.HIDDEN)
        for i in range(hidden_layers_number - 1):
            self.hidden_layers[i + 1] = Layer(layer_length[i + 1], self.hidden_layers[i].neurons, act_arr[i + 1], Type.HIDDEN)
        self.output_layer = Layer(layer_length[hidden_layers_number], self.hidden_layers[hidden_layers_number - 1].neurons, act_arr[hidden_layers_number], Type.OUTPUT)

    def network_output(self, input):
        current_input = self.input_layer.act(input)
        for i in range(self.depth):
            current_input = self.hidden_layers[i].act(current_input)
        return self.output_layer.act(current_input)



layer = Layer(3, 3, Tanh(2), Type.INPUT)
network = Network(1, 3, [3, 1], [Sigmoid(1), Tanh(2)])
print(layer.act(np.array([1, 2, 3])))
input = np.array([1, 2, 3])
print(network.network_output(input))