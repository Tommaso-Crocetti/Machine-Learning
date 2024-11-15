from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import math
import matplotlib
import pandas as pd
from ucimlrepo import fetch_ucirepo

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
            return 1 / (1 + np.exp(-(self.a * x)))
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
            self.weight_matrix = (np.random.random((self.neurons, self.weights)) - 0.5) *  0.1

    def net(self, o):
        if self.type != Type.INPUT:
            o = np.concatenate((np.array([1]), o))
        return np.dot(self.weight_matrix, o)

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
        self.store_hidden_result[0] = self.hidden_layers[0].act(current_input)
        for i in range(self.depth - 1):
            self.store_hidden_result[i + 1] = self.hidden_layers[i + 1].act(self.store_hidden_result[i])       
        return self.output_layer.act(self.store_hidden_result[self.depth - 1])
    
    #nota: LMS e backprop. tengono conto di target value monodimensionali, quindi supponiamo di avere un solo neurone di output
    def LMS(self, X, y):
        error = 0
        for index, row in X.iterrows():
            output = self.network_output(row)
            if output >= 0.5:
                discrete_output = 1
            else:
                discrete_output = 0
            target_value = y.iloc[index]
            error += np.square(target_value - discrete_output)
        return error

    def backpropagation_iteration(self, x, y):
        #calcolo dell'output continuo della rete
        output = self.network_output(x)
        #inizializzazione della matrice contenente i gradienti di tutti i pesi della rete
        store_gradient = []
        #calcolo del delta dell'output layer (in questo caso della singola output unit)
        store_output_delta = (y.iloc[0] - output) * self.output_layer.der_act(self.store_hidden_result[self.depth - 1])
        #inizializzione della matrice contenente i grandienti dei pesi dell'output layer
        current_matrix = np.zeros((self.output_layer.neurons, self.output_layer.weights))
        #aggiungo il risultato del bias al vettore degli output dell'ultimo hidden layer
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 1]))
        #inizio a iterare sugli output neruons
        for i in range(self.output_layer.neurons):
            #itero sui pesi dei singoli output neurons
            for j in range(self.output_layer.weights):
                #riempimento della matrice con i gradienti peso per peso 
                current_matrix[i][j] = store_output_delta * updated_hidden_result[j]
        #aggiunta alla matrice totale dei gradienti la matrice dei gradienti dell'output layer
        store_gradient.append(current_matrix)


        current_hidden_layer = self.hidden_layers[self.depth - 1]
        store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
        current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 2]))
        for index_neuron in range(current_hidden_layer.neurons):
            store_current_hidden_layer_delta[index_neuron] = store_output_delta * self.output_layer.weight_matrix[0][index_neuron] * (current_hidden_layer.der_act(self.store_hidden_result[self.depth - 2])[index_neuron])
            for j in range(current_hidden_layer.weights):
                current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
        store_gradient.append(current_matrix)

        next_layer_delta = store_current_hidden_layer_delta
        for hidden_layer_index in range(self.depth - 2, 0, -1):
            current_hidden_layer = self.hidden_layers[hidden_layer_index]
            store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
            current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
            updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[hidden_layer_index - 1]))
            for index_neuron in range(current_hidden_layer.neurons):
                counter = np.dot(next_layer_delta, self.hidden_layers[hidden_layer_index + 1].weight_matrix[:,index_neuron + 1])
                store_current_hidden_layer_delta[index_neuron] = counter * (current_hidden_layer.der_act(self.store_hidden_result[hidden_layer_index - 1])[index_neuron])
                for j in range(current_hidden_layer.weights):
                    current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
            next_layer_delta = store_current_hidden_layer_delta
            store_gradient.append(current_matrix)

        current_hidden_layer = self.hidden_layers[0]
        store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
        current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
        updated_hidden_result = np.concatenate((np.array([1]), x))
        for index_neuron in range(current_hidden_layer.neurons):
            counter = np.dot(next_layer_delta, self.hidden_layers[1].weight_matrix[:,index_neuron + 1])
            store_current_hidden_layer_delta[index_neuron] = counter * current_hidden_layer.der_act(x)[index_neuron]
            for j in range(current_hidden_layer.weights):
                current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
        store_gradient.append(current_matrix)
        store_gradient.reverse()
        return store_gradient

    def backpropagation_batch(self, X, y):
        batch_gradient = [np.zeros((self.hidden_layers[i].neurons, self.hidden_layers[i].weights)) for i in range(self.depth)]
        batch_gradient.append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        for index, row in X.iterrows():
            current_gradient = self.backpropagation_iteration(row, y.iloc[index])
            batch_gradient = [batch_gradient[i] - current_gradient[i] for i in range(self.depth + 1)]
        for i in range(self.depth):
            self.hidden_layers[i].weight_matrix += batch_gradient[i] * 0.05
        self.output_layer.weight_matrix += batch_gradient[self.depth] * 0.05

    def backpropagation_online(self, X, y):
        for index, row in X.iterrows():
            current_gradient = self.backpropagation_iteration(row, y.iloc[index])
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix -= current_gradient[i]
            self.output_layer.weight_matrix -= current_gradient[self.depth]

def main():

    monk_s_problems = fetch_ucirepo(id=70) 
    
    # data (as pandas dataframes) 
    X = monk_s_problems.data.features 
    y = monk_s_problems.data.targets 


    network = Network(4, X.shape[1], [4 ,10, 4, 2, 1], [Sigmoid(1), Sigmoid(1), Sigmoid(1), Sigmoid(1), Sigmoid(1), Sigmoid(1)])
    #print(network.LMS(X, y))
    #print(network.backpropagation_batch(X, y))
    #print(network.LMS(X, y))
 
    network.backpropagation_batch(X, y)
    print(network.network_output(X.iloc[0]))
    for i in range(1):
        network.backpropagation_batch(X, y)
    print(network.network_output(X.iloc[0]))
    print(network.network_output([1000,1000,1000,1000,1000,1000]))

if __name__ == "__main__":
    main()