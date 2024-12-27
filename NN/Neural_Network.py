from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import networkx as nx
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

    def __init__(self, neurons, weights, activation_class, layer_type, weight_range = 0.75):
        self.neurons = neurons
        self.type = layer_type
        self.activation_function = activation_class.activation()
        self.activation_derivate = activation_class.derivate()
        if layer_type == Type.INPUT:
            self.weights = weights
            self.weight_matrix = np.eye(neurons)
        else:
            self.weights = weights + 1
            self.weight_matrix = (np.random.uniform(-weight_range, weight_range, (self.neurons, self.weights)))

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

    def __init__(self, weight_range, hidden_layers_number, input_dimension, layer_length, activation_class_arr):
        self.depth = hidden_layers_number
        self.input_layer = Layer(input_dimension, input_dimension, Id(), Type.INPUT)
        self.hidden_layers = np.empty(self.depth, dtype=object)
        self.store_hidden_result = []
        self.hidden_layers[0] = Layer(layer_length[0], input_dimension, activation_class_arr[0], Type.HIDDEN, weight_range)
        self.store_hidden_result.append(np.zeros(self.hidden_layers[0].neurons))
        for i in range(1, hidden_layers_number):
            self.hidden_layers[i] = Layer(layer_length[i], self.hidden_layers[i - 1].neurons, activation_class_arr[i], Type.HIDDEN, weight_range)
            self.store_hidden_result.append(np.zeros(self.hidden_layers[i].neurons))
        self.output_layer = Layer(layer_length[-1], self.hidden_layers[-1].neurons, activation_class_arr[-1], Type.OUTPUT, weight_range)
        self.std_mean = {
            "X_mean": 0,
            "X_std": 1,
            "y_mean": 0,
            "y_std": 1
        }

    def set_reset(self):
        self.reset_hidden_layers = []
        for i in range(self.depth):
            self.reset_hidden_layers.append(np.copy(self.hidden_layers[i].weight_matrix))
        self.reset_output_layer = np.copy(self.output_layer.weight_matrix)

    def reset(self):
        for i in range(self.depth):
            self.hidden_layers[i].weight_matrix = np.copy(self.reset_hidden_layers[i])
        self.output_layer.weight_matrix = np.copy(self.reset_output_layer)

    def random_reset(self, weight_range = 0.5):
        for i in range(self.depth):
            current_hidden_layer = self.hidden_layers[i]
            current_hidden_layer.weight_matrix = (np.random.uniform(-weight_range, weight_range, (current_hidden_layer.neurons, current_hidden_layer.weights)))
        self.output_layer.weight_matrix = (np.random.uniform(-weight_range, weight_range, (self.output_layer.neurons, self.output_layer.weights)))

    def train_network_output(self, input):
        current_input = self.input_layer.act(input)
        self.store_hidden_result[0] = self.hidden_layers[0].act(current_input)
        for i in range(self.depth-1):
            self.store_hidden_result[i + 1] = self.hidden_layers[i + 1].act(self.store_hidden_result[i])
        return self.output_layer.act(self.store_hidden_result[self.depth - 1])

    def network_output(self, input):
        input = (input - self.std_mean["X_mean"]) / self.std_mean["X_std"]
        output = self.train_network_output(input)
        return output * self.std_mean["y_std"] + self.std_mean["y_mean"]

    def LMS_classification(self, X, y, threshold = 0.5, positive = 1, negative=0, mean = False):
        error = 0
        X_stand = self.standard(X)
        for i in range(len(X)):
            output = self.unstandard(self.network_output(X_stand.iloc[i],y))
            if output >= threshold:
                discrete_output = positive
            else:
                discrete_output = negative
            target_value = y.iloc[i]
            error += (target_value - discrete_output)**2
        return error
    
    def LMS_regression(self, X, y):
        error = 0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            error += np.dot(y.iloc[i] - output, y.iloc[i] - output)
        return error / len(X)
    
    def LED_regression(self, X, y):
        error = 0
        for i in range(len(X)):
            output = (self.network_output(X.iloc[i])*self.std_mean_arr["y_train_std"]) + self.std_mean_arr["y_train_mean"]
            error += np.sqrt(np.dot((y.iloc[i] - output), (y.iloc[i] - output)))
        return error / len(X)

    def backpropagation_batch(self, X, y, regression = True, mean = True, standardization = None, tollerance = 6, max_batches_number = 100, eta = 0.1, lambda_tichonov=0, alpha=0, validation = None, plot =False, trial = ""):
        l = len(X)
        if mean:
            eta = eta / l
            alpha = alpha / l
        if standardization:
            self.std_mean["X_mean"] = X.mean()
            self.std_mean["X_std"] = X.std()
            self.std_mean["y_mean"] = y.mean()
            self.std_mean["y_std"] = y.std()
        X_std = (X - self.std_mean["X_mean"]) / self.std_mean["X_std"]
        y_std = (y - self.std_mean["y_mean"]) / self.std_mean["y_std"]
        error_function = self.LED_regression if regression else self.LMS_classification
        task_errors = []
        LMS_errors = []
        task_validation_errors = []
        LMS_validation_errors = []
        batch_gradient = [[np.zeros((self.hidden_layers[i].neurons, self.hidden_layers[i].weights)) for i in range(self.depth)]for i in range(2)]
        batch_gradient[0].append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        batch_gradient[1].append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        for i in range(max_batches_number):
#             print(f"Iterazione {i}")
            task_errors.append(error_function(X, y))   
            LMS_errors.append(self.LMS_regression(X, y)) 
            if validation: 
                task_validation_errors.append(error_function(validation[0], validation[1]))
                LMS_validation_errors.append(self.LMS_regression(validation[0], validation[1]))
            if len(LMS_errors) > 1:
                if np.abs(LMS_errors[-1] - LMS_errors[-2]) < 10**-tollerance:
                    print(f"Convergenza raggiunta in {i} iterazioni, {np.abs(LMS_errors[-1] - LMS_errors[-2])}")
                    break
                if LMS_errors[-1] - LMS_errors[-2] > 10**tollerance:
                    print(f"Errore crescente alla iterazione {i}, {LMS_errors[-1] - LMS_errors[-2]}")
                    break
            for j in range(self.depth):
                batch_gradient[1][j] = lambda_tichonov * self.hidden_layers[j].weight_matrix
                self.hidden_layers[j].weight_matrix += (alpha * batch_gradient[0][j])
            batch_gradient[1][-1] = lambda_tichonov * self.output_layer.weight_matrix
            self.output_layer.weight_matrix += (alpha * batch_gradient[0][-1])
            batch_gradient[0] = self.backpropagation_iteration(X_std.iloc[0], y_std.iloc[0])
            for i in range(1, l):
                current_gradient = self.backpropagation_iteration(X_std.iloc[i], y_std.iloc[i])
                batch_gradient[0] = [batch_gradient[0][i] + current_gradient[i] for i in range(self.depth + 1)]
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix += (eta * batch_gradient[0][i]) - batch_gradient[1][i]
            self.output_layer.weight_matrix += (eta * batch_gradient[0][-1]) - batch_gradient[1][-1]
        if plot:
            self.plot_error(LMS_errors, LMS_validation_errors, str(trial) + "LMS_error")
            self.plot_error(task_errors, task_validation_errors, str(trial) + "Task_error")
            
    def backpropagation_online(self, X, y):
        for index, row in X.iterrows():
            current_gradient = self.backpropagation_iteration(row, y.iloc[index])
            for i in range(self.depth):
                self.hidden_layers[i].weight_matrix += current_gradient[i]
            self.output_layer.weight_matrix += current_gradient[self.depth]

    def backpropagation_iteration(self, x, y):
        output = self.train_network_output(x)
        store_gradient = []
        current_layer_delta = np.zeros(self.output_layer.neurons)
        current_layer_der = self.output_layer.der_act(self.store_hidden_result[-1])
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 1]))
        current_layer_delta = (y - output) * current_layer_der
        store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
#         print("Output layer delta: ", current_layer_delta)
        next_layer_delta = current_layer_delta
        if (self.depth == 1):
            current_layer = self.hidden_layers[0]
            current_layer_delta = np.zeros(current_layer.neurons)
            current_layer_der = current_layer.der_act(x)
            updated_hidden_result = np.concatenate((np.array([1]), x))
            for index_neuron in range(current_layer.neurons):
                current_layer_delta[index_neuron] = np.dot(next_layer_delta, self.output_layer.weight_matrix[:,index_neuron + 1]) * current_layer_der[index_neuron]
            store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
            store_gradient.reverse()
            return store_gradient
        current_layer = self.hidden_layers[-1]
        current_layer_delta = np.zeros(current_layer.neurons)
        current_layer_der = current_layer.der_act(self.store_hidden_result[-2])
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[-2]))
        for index_neuron in range(current_layer.neurons):
            current_layer_delta[index_neuron] = np.dot(next_layer_delta, self.output_layer.weight_matrix[:,index_neuron + 1]) * current_layer_der[index_neuron]
        store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
#         print("Last hidden layer delta: ", current_layer_delta)
        next_layer_delta = current_layer_delta
        for hidden_layer_index in range(self.depth - 2, 0, -1):
            current_layer = self.hidden_layers[hidden_layer_index]
            current_layer_delta = np.zeros(current_layer.neurons)
            current_layer_der = current_layer.der_act(self.store_hidden_result[hidden_layer_index - 1])
            updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[hidden_layer_index - 1]))
            for index_neuron in range(current_layer.neurons):
                current_layer_delta[index_neuron] = np.dot(next_layer_delta, self.hidden_layers[hidden_layer_index + 1].weight_matrix[:,index_neuron + 1]) * current_layer_der[index_neuron]
            store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
#             print(f"Hidden layer {hidden_layer_index} delta: ", current_layer_delta)
            next_layer_delta = current_layer_delta
        current_layer = self.hidden_layers[0]
        current_layer_delta = np.zeros(current_layer.neurons)
        current_layer_der = current_layer.der_act(x)
        updated_hidden_result = np.concatenate((np.array([1]), x))
        for index_neuron in range(current_layer.neurons):
            current_layer_delta[index_neuron] = np.dot(next_layer_delta, self.hidden_layers[1].weight_matrix[:,index_neuron + 1]) * current_layer_der[index_neuron]
        store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
#         print("First hidden layer delta: ", current_layer_delta)
        store_gradient.reverse()
        return store_gradient
    
    def grid_search(self, training_data, validation_data, eta_range, lambda_tichonov_range, alpha_range, net_name):
        best_comb = -1
        best_model = [0, 0, 0, 0]
        best_validation_error = np.inf
        counter = 0
        self.set_reset()
        for current_eta in eta_range:
            for current_lambda_tichonov in lambda_tichonov_range:
                for current_alpha in alpha_range:
#                     print(f"Numero di neuroni: {current_hidden_units_number}")
#                     print(f"Eta: {current_eta}")
#                     print(f"Lambda: {current_lambda_tichonov}")
#                     print(f"Alpha: {current_alpha}")
                    self.backpropagation_batch(training_data[0], training_data[1], standardization = True, tollerance = 3, max_batches_number = 300, eta = 10**current_eta, lambda_tichonov=10**current_lambda_tichonov, alpha = 10**current_alpha, validation = validation_data, plot = True, trial = counter)
                    current_validation_error = self.LED_regression(validation_data[0], validation_data[1])
                    print(f"{counter}, Errore di validazione: {current_validation_error}")
                    if current_validation_error < best_validation_error:
                        best_comb = counter
                        best_model = [counter, current_eta, current_lambda_tichonov, current_alpha]
                        best_validation_error = current_validation_error
                    counter += 1
                    self.reset()
        self.write_result(net_name, best_comb, best_model, best_validation_error)
        return best_comb, best_model, best_validation_error
    
    def plot_error(self, errors, validation_errors, filename):
        plot = plt.figure(figsize=(8, 6))
        plt.plot(range(len(errors)), errors, c='blue', label='Training Error')
        plt.plot(range(len(validation_errors)), validation_errors, c='red', label='Test Error')
        plt.title(filename)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.grid()
        plt.legend()
        plt.savefig("Plot/Png/" + filename + ".png")
        with open("Plot/Pickle/" + filename + ".pkl", "wb") as f:
            pickle.dump(plot, f)
        plt.show()
        plt.close()
    
    def plot_output(self, X, y, filename = ""):
        data = []
        for i in range(len(X)):
            output = pd.Series(self.network_output(X.iloc[i]))
            data.append({
                'target_x': output.iloc[0],
                'target_y': output.iloc[1],
                'target_z': output.iloc[2],
            })
        data = pd.DataFrame(data)
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')  


        ax.scatter(y['target_x'], y['target_y'], y['target_z'], c='blue', marker='o', label="Target value", s=50)  
        ax.scatter(data['target_x'], data['target_y'], data['target_z'], c='red', marker='o', label="Network output", s=50)  

        plt.legend()


        ax.set_title(filename)
        ax.set_xlabel('Asse X')
        ax.set_ylabel('Asse Y')
        ax.set_zlabel('Asse Z')

        x_min = min(y['target_x'].min(), data['target_x'].min())
        x_max = max(y['target_x'].max(), data['target_x'].max())
        y_min = min(y['target_y'].min(), data['target_y'].min())
        y_max = max(y['target_y'].max(), data['target_y'].max())
        z_min = min(y['target_z'].min(), data['target_z'].min())
        z_max = max(y['target_z'].max(), data['target_z'].max())

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        with open("Plot/Pickle/" + filename + ".pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.tight_layout()
        plt.show()

    def plot(self, filename):
        
        def weight_to_color (x):
            return [0, 0, 1] if x < 0 else [1, 0, 0]


        def weight_alpha(x): 
            alpha = np.abs(np.divide(x,2))
            return alpha

        G = nx.DiGraph()
        layers = []
        layers.append(self.input_layer.neurons)
        for i in range(self.depth):
            layers.append(self.hidden_layers[i].neurons)
        layers.append(self.output_layer.neurons)

        pos = {}
        node_labels = {}
        node_colors = {}
        node_sizes = {}
        y_offset = 0
        max_layers = np.max(layers[1:])

        for layer, n_units in enumerate(layers):
            if layer != len(layers) - 1:
                y_offset = -(n_units/2)
                node_id_minus_one = f"L{layer}_N{-1}"
                G.add_node(node_id_minus_one)  
                node_sizes [node_id_minus_one] = 200
                node_labels[node_id_minus_one] = " "
                node_colors[node_id_minus_one] = "yellow"
                pos[node_id_minus_one] = (layer / 2, - (max_layers/2+1))

            for unit in range(n_units):
                node_id = f"L{layer}_N{unit}"
                G.add_node(node_id)
                if layer == 0:
                    y_offset = -((n_units-1))/2
                    pos[node_id] = (layer / 2, y_offset + unit)
                    node_sizes [node_id] = 200
                    node_labels[node_id] = " "  
                    node_colors[node_id] = "gray"
                elif layer < len(layers)-1: 
                    y_offset = -((n_units-1))/2
                    pos[node_id] = (layer / 2, y_offset + unit)
                    node_sizes [node_id] = 200
                    node_labels[node_id] = " "  
                    node_colors[node_id] = "green"
                else:
                    y_offset = -((n_units-1))/2
                    pos[node_id] = (layer / 2, y_offset + unit)
                    node_sizes [node_id] = 200
                    node_labels[node_id] = " "
                    node_colors[node_id] = "green"


        for layer in range(self.depth+1):
            for src in range(-1, layers[layer]):
                for dest in range(layers[layer + 1]):
                    src_id = f"L{layer}_N{src}"
                    dest_id = f"L{layer + 1}_N{dest}"
                    if (layer < self.depth):
                        G.add_edge(src_id, dest_id)
                        G[src_id][dest_id]["weight"] = self.hidden_layers[layer].weight_matrix[dest][src]
                    elif (layer == ((self.depth))):
                        G.add_edge(src_id, dest_id)
                        G[src_id][dest_id]["weight"] = self.output_layer.weight_matrix[dest][src]


        edge_colors = []
        for u, v in G.edges():
            weight = G[u][v]["weight"] 
            color = weight_to_color(weight)
            alpha = weight_alpha(weight)
            edge_colors.append((color[0], color[1], color[2], alpha)) 


        node_colors_list = np.array(list(node_colors.values()))
        node_sizes_list = np.array(list(node_sizes.values()))
        fig = plt.figure(figsize=(16, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=node_sizes_list, edgecolors="black")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=3, arrows=False)

        plt.axis('off')
        plt.title("Neural Network")
        with open("Plot/Pickle/" + filename + ".pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.tight_layout()

        plt.show()

    def write_result(self, net_name, best_comb, best_model, best_validation_error):
        with open(net_name + "_grid_search.txt", "a") as f:
            f.write(f"\n|\t{best_comb}\t\t|\t{best_model}\t|\t{best_validation_error}\t|")
        f.close()