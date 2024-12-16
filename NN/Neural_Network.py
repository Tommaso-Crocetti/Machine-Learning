from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import math
from NN.NN_graph import plot_neural_network
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo
import pickle

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

class Relu(Function):

    def activation(self):
        def relu(x):
            return np.max(x,0)
        return relu

    def derivate(self):
        def relu_der(x):
            if (x < 0): return 0
            return 1
        return relu_der

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

    def __init__(self, neurons, weights, activation_class, layer_type, weight_scaling = 1):
        self.neurons = neurons
        self.type = layer_type
        self.activation_function = activation_class.activation()
        self.activation_derivate = activation_class.derivate()
        if layer_type == Type.INPUT:
            self.weights = weights
            self.weight_matrix = np.eye(neurons)
        else:
            self.weights = weights + 1
            self.weight_matrix = (np.random.uniform(-weight_scaling, weight_scaling, (self.neurons, self.weights)))

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

    def __init__(self, weigth_scaling, hidden_layers_number, input_dimension, layer_length, activation_class_arr):
        self.depth = hidden_layers_number
        self.store_hidden_result = [] 
        self.input_layer = Layer(input_dimension, input_dimension, Id(), Type.INPUT)
        self.hidden_layers = np.empty(self.depth, dtype=object)
        self.hidden_layers[0] = Layer(layer_length[0], input_dimension, activation_class_arr[0], Type.HIDDEN, weigth_scaling)
        self.store_hidden_result.append(np.zeros(self.hidden_layers[0].neurons))
        for i in range(hidden_layers_number - 1):
            self.hidden_layers[i + 1] = Layer(layer_length[i + 1], self.hidden_layers[i].neurons, activation_class_arr[i + 1], Type.HIDDEN, weigth_scaling)
            self.store_hidden_result.append(np.zeros(self.hidden_layers[i + 1].neurons))
        self.output_layer = Layer(layer_length[hidden_layers_number], self.hidden_layers[hidden_layers_number - 1].neurons, activation_class_arr[hidden_layers_number], Type.OUTPUT, weigth_scaling)

    def plot_error(self, errors, filename):
        plot = plt.figure(figsize=(8, 6))
        plt.plot(range(len(errors)), errors, marker='o', label='Errore LMS')
        plt.title("Curva dell'Errore LMS all'aumentare delle epoche")
        plt.xlabel("Epoche")
        plt.ylabel("Errore LMS")
        plt.grid()
        plt.legend()
        plt.savefig("Plot/Png/" + filename + ".png")
        with open("Plot/Pickle/" + filename + ".pkl", "wb") as f:
            pickle.dump(plot, f)
        plt.show()
        plt.close()
    
    def plot_target(self, y, filename):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')  # Grafico 3D

        # Plot dei punti
        ax.scatter(y['target_x'], y['target_y'], y['target_z'], c='red', marker='o', s=50)  # Scatter plot

        # Personalizzazione
        ax.set_title('Punti 3D da Dataset Pandas')
        ax.set_xlabel('Asse X')
        ax.set_ylabel('Asse Y')
        ax.set_zlabel('Asse Z')

        # Mostra il grafico
        plt.show()

    def plot_from(self,x):
        result = self.network_output(x)
        plot_neural_network(self, x, result)

    def network_output(self, input):
        current_input = self.input_layer.act(input)
        self.store_hidden_result[0] = self.hidden_layers[0].act(current_input)
        for i in range(self.depth-1):
            self.store_hidden_result[i + 1] = self.hidden_layers[i + 1].act(self.store_hidden_result[i])
        return self.output_layer.act(self.store_hidden_result[self.depth - 1])
    
    def plot_output(self, X, filename):
        df = pd.DataFrame(columns=['target_x', 'target_y', 'target_z'])
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            df = df._append({
                'target_x': output[0],
                'target_y': output[1],
                'target_z': output[2],
            }, ignore_index=True)
        self.plot_target(df, "")


    #nota: LMS e backprop. tengono conto di target value monodimensionali, quindi supponiamo di avere un solo neurone di output
    def LMS_classification(self, X, y, threshold=0.5, positive=1, negative=0, mean = False):
        error = 0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            if output >= threshold:
                discrete_output = positive
            else:
                discrete_output = negative
            target_value = y.iloc[i]
            error += (target_value - discrete_output)**2
        if mean: 
            return (error / len(X))
        return error
    
    def LMS_regression(self,X,y, mean = False):
        error=0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            error += np.dot(y.iloc[i] - output, y.iloc[i] - output)
        if mean:
            return (error / len(X))
        return error
    
    def LED_regression(self, X, y, mean = False):
        error = 0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            error += np.sqrt(np.dot((y.iloc[i] - output), (y.iloc[i] - output)))
        if mean:
            return (error / len(X))
        return error

    def backpropagation_batch(self, X, y, regression = True, batches_number=100, eta=0.1, lambda_tichonov=0, alpha=0, validation = None, plot=False):
        errors = []
        validation_errors = []
        #inizializzo la matrice che conterrà la somma di tutte le matrici store_gradient per ogni hidden layer
        batch_gradient = [[np.zeros((self.hidden_layers[i].neurons, self.hidden_layers[i].weights)) for i in range(self.depth)]for i in range(2)]
        #aggiungo l'ultimo pezzo di batch_gradient che conterrà la somma di tutti i gradienti per l'output layer
        batch_gradient[0].append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        batch_gradient[1].append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        for i in range(batches_number):
            print(f"Iterazione {i}")
            if regression:
                errors.append(self.LMS_regression(X, y, True))
            else: 
                errors.append(self.LMS_classification(X, y))    
            if validation:
                if regression:
                    validation_errors.append(self.LMS_regression(validation[0], validation[1], True))
                else: 
                    validation_errors.append(self.LMS_classification(validation[0], validation[1]))
            #itero sul dataset
            for j in range(self.depth):
                batch_gradient[1][j] = lambda_tichonov * self.hidden_layers[j].weight_matrix
                self.hidden_layers[j].weight_matrix += alpha * batch_gradient[0][j]
            batch_gradient[1][-1] = lambda_tichonov * self.output_layer.weight_matrix
            self.output_layer.weight_matrix += alpha * batch_gradient[0][-1]
            for i in range(len(X)):
                #calcolo store_gradient per il pattern corrente con il suo target
                current_gradient = self.backpropagation_iteration(X.iloc[i], y.iloc[i])
                #aggiungo il gradiente appena calcolato 
                batch_gradient[0] = [batch_gradient[0][i] + current_gradient[i] for i in range(self.depth + 1)]
            #per ogni hidden layer aggiorno i pesi sommando la matrice batch
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix += (batch_gradient[0][i] * eta / len(X)) - batch_gradient[1][i]
            self.output_layer.weight_matrix += (batch_gradient[0][self.depth] * eta / len(X)) - batch_gradient[1][-1]
        if plot:
            self.plot_error(errors, "training_error")
            if validation: 
                self.plot_error(validation_errors, "validation_error")
            
    def backpropagation_online(self, X, y):
        for index, row in X.iterrows():
            current_gradient = self.backpropagation_iteration(row, y.iloc[index])
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix += current_gradient[i]
            self.output_layer.weight_matrix += current_gradient[self.depth]

    def backpropagation_iteration(self, x, y):
        #calcolo dell'output continuo della rete
        output = self.network_output(x)
        #inizializzazione della matrice contenente i gradienti di tutti i pesi della rete
        store_gradient = []
        store_output_delta = []
        #calcolo del delta dell'output layer (in questo caso della singola output unit)
        for i in range(self.output_layer.neurons):
            store_output_delta.append((y.iloc[i] - output[i]) * self.output_layer.der_act(self.store_hidden_result[self.depth-1])[i])
        #inizializzione della matrice contenente i grandienti dei pesi dell'output layer
        current_matrix = np.zeros((self.output_layer.neurons, self.output_layer.weights))
        #aggiungo il risultato del bias al vettore degli output dell'ultimo hidden layer
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 1]))
        #inizio a iterare sugli output neruons
        for i in range(self.output_layer.neurons):
            #itero sui pesi dei singoli output neurons
            for j in range(self.output_layer.weights):
                #riempimento della matrice con i gradienti peso per peso 
                current_matrix[i][j] = store_output_delta[i] * updated_hidden_result[j]
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
                counter = np.dot(store_output_delta[0], self.output_layer.weight_matrix[:,index_neuron + 1])
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
        for output_index in range(self.output_layer.neurons):
            for index_neuron in range(current_hidden_layer.neurons):
                #nella matrice contenente i delta di questo layer metto il prodotto del delta dell'output layer (in questo caso un solo neurone) 
                #per il peso che va da index_neuron all'output neuron per la derivata della funzione di attivazione applicata alla net del neurone
                store_current_hidden_layer_delta[index_neuron] = np.dot(store_output_delta, self.output_layer.weight_matrix[:,index_neuron+1]) * (current_hidden_layer.der_act(self.store_hidden_result[self.depth - 2])[index_neuron])
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