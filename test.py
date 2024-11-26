from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import math
from NN_graph import plot_neural_network
import matplotlib.pyplot as plt
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

    def activation(self):
        def tanh(x):
            return np.tanh(self.a*x/2)
        return tanh
    
    def derivate(self):
        def tanh_der(x):
            return 1 - (self.activation()(x))**2
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
            self.weight_matrix = (np.random.random((self.neurons, self.weights)) - 0.5) *  0.9

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

    def plot(self, errors):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(errors)), errors, marker='o', label='Errore LMS')
        plt.title("Curva dell'Errore LMS all'aumentare delle epoche")
        plt.xlabel("Epoche")
        plt.ylabel("Errore LMS")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_from(self,x):
        result = self.network_output(x)
        #sotto considero l'array di result perché per ora il result è un solo valore, poi sarà
        #un array di valori    
        plot_neural_network(self, x, result)

    def network_output(self, input):
        current_input = self.input_layer.act(input)
        self.store_hidden_result[0] = self.hidden_layers[0].act(current_input)
        for i in range(self.depth-1):
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
            error += (target_value - discrete_output)**2
        return error

    def backpropagation_iteration(self, x, y):
        #calcolo dell'output continuo della rete
        output = self.network_output(x)
        #inizializzazione della matrice contenente i gradienti di tutti i pesi della rete
        store_gradient = []
        #calcolo del delta dell'output layer (in questo caso della singola output unit)
        store_output_delta = (y - output) * self.output_layer.der_act(self.store_hidden_result[self.depth-1])
        #inizializzione della matrice contenente i grandienti dei pesi dell'output layer
        current_matrix = np.zeros((self.output_layer.neurons, self.output_layer.weights))
        #aggiungo il risultato del bias al vettore degli output dell'ultimo hidden layer
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 1]))
        #inizio a iterare sugli output neruons
        for i in range(self.output_layer.neurons):
            #itero sui pesi dei singoli output neurons
            for j in range(self.output_layer.weights):
                #riempimento della matrice con i gradienti peso per peso 
                current_matrix[i][j] = store_output_delta[0] * updated_hidden_result[j]
        #aggiunta alla matrice totale dei gradienti la matrice dei gradienti dell'output layer
        store_gradient.append(current_matrix)

        if (self.depth == 1):
            current_hidden_layer = self.hidden_layers[0]
            store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
            #inizializzo la matrice che conterrà i gradienti dei pesi dell' hidden layer più vicino all'output layer
            current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
            updated_hidden_result = np.concatenate((np.array([1]), x))
            #itero sui neuroni del layer corrente
            for index_neuron in range(current_hidden_layer.neurons):
                #calcolo il prodotto scalare tra il vettore contenente il delta del layer più a destra e il vettore contenente i pesi 
                #di ogni neurone del layer più a destra che li collegano all'index_neuronesimo neurone
                counter = np.dot(store_output_delta, self.output_layer.weight_matrix[:,index_neuron + 1])
                #aggiungo alla matrice contenente i delta del layer corrente il prodotto di counter e la derivata della funzione di attivazione
                #applicata alla net delle uscite dei neuroni precedenti
                store_current_hidden_layer_delta[index_neuron] = counter * current_hidden_layer.der_act(x)[index_neuron]
                #itero sui pesi dei singoli nueroni del layer corrente
                for j in range(current_hidden_layer.weights):
                    #aggiungo alla matrice il prodotto del delta del neurone corrente per l'uscita del j-esimo neurone
                    current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
                #aggiungo la matrice appena calcolata alla matrice totale
            store_gradient.append(current_matrix)
            #inverto l'ordine della matrice contenente i gradienti di ogni peso in modo da avere prima i gradienti del primo hidden layer
            # e dopo i gradienti dell'output layer 
            store_gradient.reverse()
            return store_gradient

        #passiamo a valutare i gradienti dell' hidden layer più vicino all'output layer
        current_hidden_layer = self.hidden_layers[self.depth - 1]
        #inizializzo il vettore contenente i delta dell' hidden layer più vicino all'output layer
        store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
        #inizializzo la matrice che conterrà i gradienti dei pesi dell' hidden layer più vicino all'output layer
        current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
        #aggiungo il bias
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 2]))
        #itero sui neuroni del layer
        for index_neuron in range(current_hidden_layer.neurons):
            #nella matrice contenente i delta di questo layer metto il prodotto del delta dell'output layer (in questo caso un solo neurone) 
            #per il peso che va da index_neuron all'output neuron per la derivata della funzione di attivazione applicata alla net del neurone
            store_current_hidden_layer_delta[index_neuron] = store_output_delta * self.output_layer.weight_matrix[0][index_neuron+1] * (current_hidden_layer.der_act(self.store_hidden_result[self.depth - 2])[index_neuron])
            #itero sul numero di pesi del layer corrente
            for j in range(current_hidden_layer.weights):
                #aggiungo il prodotto del delta del index_neuron per il risultato del j-esimo neurone al layer precedente (quello verso l'input layer)
                current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
            #aggiungo la matrice totale con l'aggiornamento dei pesi di questo layer
        store_gradient.append(current_matrix)

        #aggiorniamo il valore del delta del prossimo layer con il valore del delta dell'ultimo hidden layer
        next_layer_delta = store_current_hidden_layer_delta

        #comincio a iterare partendo dal penultimo hidden layer fino ad arrivare al secondo hidden layer con passo -1
        for hidden_layer_index in range(self.depth - 2, 0, -1):
            #setto l'hidden layer corrente
            current_hidden_layer = self.hidden_layers[hidden_layer_index]
            #inizializzo l'array che conterrà i delta dei neuroni di questo layer 
            store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
            #inizializzo la matrice che conterrà gli aggiornamenti dei pesi dei neuroni di questo layer
            current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
            #aggiungo il bias ai neuroni del layer più vicino all'input layer
            updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[hidden_layer_index - 1]))
            #itero sul numero di neuroni di questo layer
            for index_neuron in range(current_hidden_layer.neurons):
                #calcolo il prodotto scalare tra il vettore dei delta dei neuroni del layer più a destra per il vettore contenente i pesi 
                #di ogni neurone del layer più a destra che li collegano all'index_neuronesimo neurone
                counter = np.dot(next_layer_delta, self.hidden_layers[hidden_layer_index + 1].weight_matrix[:,index_neuron + 1])
                #calcolo il prodotto di counter per la derivata della funzione di attivazione del layer corrente applicata
                #ai risultati del layer più a sinistra
                store_current_hidden_layer_delta[index_neuron] = counter * (current_hidden_layer.der_act(self.store_hidden_result[hidden_layer_index - 1])[index_neuron])
                #itero sui singoli pesi del layer corrente
                for j in range(current_hidden_layer.weights):
                    #aggiungo alla matrice che contiene gli aggiornamenti dei pesi il gradiente relativo al j-esimo peso dell'index_neuresimo neurone
                    current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
            
            #aggiorno il next_layer_delta in modo che il delta del layer corrente valga come delta del layer successivo per il layer
            #che considero alla prossima iterazione 
            next_layer_delta = store_current_hidden_layer_delta
            #aggiungo la matrice che contiene gli aggiornamenti dei pesi di questo layer alla matrice totale
            store_gradient.append(current_matrix)

        #considero l'ultimo hidden layer, ossia quello a sinistra dell'input layer
        current_hidden_layer = self.hidden_layers[0]
        #inizializzo la matrice che conterrà i delta di questo layer
        store_current_hidden_layer_delta = np.zeros(current_hidden_layer.neurons)
        #inizializzo la matrice che conterrà gli aggiornamenti dei pesi di questo layer
        current_matrix = np.zeros((current_hidden_layer.neurons, current_hidden_layer.weights))
        #aggiungo il bias ai neuroni del layer più a sinistra (l'input layer)
        updated_hidden_result = np.concatenate((np.array([1]), x))
        #itero sui neuroni del layer corrente
        for index_neuron in range(current_hidden_layer.neurons):
            #calcolo il prodotto scalare tra il vettore contenente il delta del layer più a destra e il vettore contenente i pesi 
            #di ogni neurone del layer più a destra che li collegano all'index_neuronesimo neurone
            counter = np.dot(next_layer_delta, self.hidden_layers[1].weight_matrix[:,index_neuron + 1])
            #aggiungo alla matrice contenente i delta del layer corrente il prodotto di counter e la derivata della funzione di attivazione
            #applicata alla net delle uscite dei neuroni precedenti
            store_current_hidden_layer_delta[index_neuron] = counter * current_hidden_layer.der_act(x)[index_neuron]
            #itero sui pesi dei singoli nueroni del layer corrente
            for j in range(current_hidden_layer.weights):
                #aggiungo alla matrice il prodotto del delta del neurone corrente per l'uscita del j-esimo neurone
                current_matrix[index_neuron][j] = store_current_hidden_layer_delta[index_neuron] * updated_hidden_result[j]
            #aggiungo la matrice appena calcolata alla matrice totale
        store_gradient.append(current_matrix)
        #inverto l'ordine della matrice contenente i gradienti di ogni peso in modo da avere prima i gradienti del primo hidden layer
        # e dopo i gradienti dell'output layer 
        store_gradient.reverse()
        return store_gradient

    def backpropagation_batch(self, X, y, batches_number,eta=0.1, lambda_tikonov=0,alpha=0,plot=False):
        errors = []
        for i in range(batches_number):
            errors.append(self.LMS(X, y))
            #inizializzo la matrice che conterrà la somma di tutte le matrici store_gradient per ogni hidden layer
            batch_gradient = [np.zeros((self.hidden_layers[i].neurons, self.hidden_layers[i].weights)) for i in range(self.depth)]
            #aggiungo l'ultimo pezzo di batch_gradient che conterrà la somma di tutti i gradienti per l'output layer
            batch_gradient.append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
            for j in range(self.depth):
                self.hidden_layers[j].weight_matrix += alpha*batch_gradient[j]
            self.output_layer.weight_matrix += alpha*batch_gradient[-1]
            #itero sul dataset
            for index, row in X.iterrows():
                #calcolo store_gradient per il pattern corrente con il suo target
                current_gradient = self.backpropagation_iteration(row, y.iloc[index])
                #aggiungo il gradiente appena calcolato 
                batch_gradient = [batch_gradient[i] + current_gradient[i] for i in range(self.depth + 1)]
            #per ogni hidden layer aggiorno i pesi sommando la matrice batch
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix += (batch_gradient[i] * eta) - (lambda_tikonov * self.hidden_layers[i].weight_matrix)
            self.output_layer.weight_matrix += (batch_gradient[self.depth] * eta) - (lambda_tikonov * self.output_layer.weight_matrix)

        if plot:
            self.plot(errors)
            
    def backpropagation_online(self, X, y):
        for index, row in X.iterrows():
            current_gradient = self.backpropagation_iteration(row, y.iloc[index])
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix += current_gradient[i]
            self.output_layer.weight_matrix += current_gradient[self.depth]

def main():

    monk_s_problems = pd.read_csv("./monks-3.test",sep = "\s+", header=None) 
    
    # data (as pandas dataframes) 
    n_colonne = monk_s_problems.shape[1]
    colonne= ['target'] + [ f'featuer{i}' for i in range(1,n_colonne -1)] + ['datanumber']
    monk_s_problems.columns = colonne

    X = monk_s_problems.drop(columns=["target", 'datanumber'])
    y = monk_s_problems["target"] 

    X_encoded = pd.get_dummies(X, dtype=float, columns=[f'featuer{i}' for i in range(1,n_colonne -1)])

    network = Network(1, X_encoded.shape[1], [2,1], [Sigmoid(1),Sigmoid(1)])

    #network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    network.backpropagation_batch(X_encoded, y, batches_number = 200,eta = 0.01, lambda_tikonov= 0.002, plot = True)
    
    print(network.LMS(X_encoded, y))
    
    network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])



if __name__ == "__main__":
    main()