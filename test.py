from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import matplotlib
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Definire un'interfaccia chiamata "Figura"
class Function(ABC):
    
    @abstractmethod
    def activation(self):
        pass
    
    @abstractmethod
    def derivate(self):
        pass


class Id(Function):

    def activation(self):
        def id(x):
            return x
        return id

    def derivate(self):
        def id_der(x):
            return 1
        return id_der
    
class Sigmoid(Function):

    def __init__(self, a):
        self.a = a 

    def activation(self):
        def sigmoid(x):
            return 1 / (1 + np.exp((self.a * -x)))
        return sigmoid

    def derivate(self):
        def sigmoid_der(x):
            return self.activation()(x)*(1-self.activation()(x))
        return sigmoid_der

class Tanh(Function):

    def __init__(self, a):
        self.a = a
        self.sigmoid = Sigmoid(self.a).activation()

    def activation(self):
        def tanh(x):
            return np.sum((2 * self.sigmoid(x)), -1)
        return tanh
    
    def derivate(self):
        def tanh_der(x):
            return 1 - np.square(self.activation()(x))
        return tanh_der


class Type(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:

    def __init__(self, neurons, weights, activation_class, layer_type):
        self.neurons = neurons
        self.type = layer_type
        self.activation_function = activation_class.activation()
        self.activation_derivate = activation_class.derivate()
        if layer_type == Type.INPUT:
            self.weights = weights
            self.weight_matrix = np.eye(neurons)
        else:
            self.weights = weights + 1
            self.weight_matrix = np.random.random((self.neurons, self.weights))

    def net(self, o):
        if self.type != Type.INPUT:
            o = np.concatenate((np.array([1]), o))
        return np.dot(self.matrix, o)

    def act(self, o):
        f = np.vectorize(self.activation_function)
        return f(self.net(o))

    def der_act(self, o):
        f = np.vectorize(self.activation_derivate)
        return f(self.net(o))


class Network:

    def __init__(self, hidden_layers_number, input_dimension, layer_length, activation_class_arr):
        self.depth = hidden_layers_number
        self.store_hidden_result = [] 
        self.input_layer = Layer(input_dimension, input_dimension, Id(), Type.INPUT)
        self.hidden_layers = np.empty(self.depth, dtype=object)
        self.hidden_layers[0] = Layer(layer_length[0], input_dimension, activation_class_arr[0], Type.HIDDEN)
        self.store_hidden_result.append(np.zeros(self.hidden_layers[0].neurons))
        for i in range(hidden_layers_number - 1):
            self.hidden_layers[i + 1] = Layer(layer_length[i + 1], self.hidden_layers[i].neurons, activation_class_arr[i + 1], Type.HIDDEN)
            self.store_hidden_result.append(np.zeros(self.hidden_layers[i + 1].neurons))
        self.output_layer = Layer(layer_length[hidden_layers_number], self.hidden_layers[hidden_layers_number - 1].neurons, activation_class_arr[hidden_layers_number], Type.OUTPUT)

    def network_output(self, input):
        current_input = self.input_layer.act(input)
        self.store_hidden_result[0] = self.hidden_layers[i].act(current_input)
        for i in range(self.depth - 1):
            self.store_hidden_result[i + 1] = self.hidden_layers[i].act(self.store_hidden_result[i])
        return self.output_layer.act(self.store_hidden_result[self.depth - 1])
    
    #nota: LMS e backprop. tengono conto di target value monodimensionali, quindi supponiamo di avere un solo neurone di output
    def LMS(self, X, y):
        error = 0
        for index, row in X.iterrows():
            output = self.network_output(row)
            target_value = y.iloc[index]
            error += np.square(target_value - output)
        return error

    def backpropagation(self, X, y):
        for index, row in X.iterrows():
            output = self.network_output(row);
            store_gradient = []
            store_output_delta = (y.iloc[index] - output) * self.output_layer.der_act(self.store_hidden_result[self.depth - 1])
            for i in range(self.output_layer.neurons):
                store_gradient.append([])
                for j in range(self.output_layer.weights):
                    store_gradient[0].append(store_output_delta * self.store_hidden_result[self.depth - 1][j])
            # array di array, contiene i delta dei neuroni degli hidden layer
            store_hidden_layers_delta = []
            current_hidden_layer = self.hidden_layers[self.depth - 1]
            current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
            for index_neuron in range(current_hidden_layer.neurons):
                current_hidden_layer_delta[index_neuron] = store_output_delta * self.output_layer.weight_matrix[0][index_neuron] * current_hidden_layer.der_act(self.store_hidden_result[self.depth - 2])
                store_gradient.append([])
                for j in range(self.current_hidden_layer.weights):
                    store_gradient[1].append(current_hidden_layer_delta[index_neuron] * self.store_hidden_result[self.depth - 2][j])

    
def main():

    monk_s_problems = fetch_ucirepo(id=70) 
    
    # data (as pandas dataframes) 
    X = monk_s_problems.data.features 
    y = monk_s_problems.data.targets 

    layer = Layer(3, 3, Tanh(2), Type.INPUT)
    network = Network(2, X.shape[1], [3, 3, 1], [Sigmoid(1), Sigmoid(1), Tanh(2)])
    print(layer.act(np.array([1, 2, 3])))
    input = np.array([1, 2, 3])
    print(network.network_output(X.iloc[0]))
    print(network.LMS(X, y).values)

if __name__ == "__main__":
    main()