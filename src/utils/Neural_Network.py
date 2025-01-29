from abc import ABC, abstractmethod
from enum import Enum
from utils.plot import *

class Function(ABC):
    
    @abstractmethod
    def activation(self):
        pass
    
    @abstractmethod
    def derivate(self):
        pass

class Id(Function):

    @staticmethod
    def id(x):
        return x

    @staticmethod
    def id_der(x):
        return 1

    def activation(self):
        return self.id

    def derivate(self):
        return self.id_der

class Relu(Function):

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_der(x):
        return 1 if x > 0 else 0

    def activation(self):
        return self.relu

    def derivate(self):
        return self.relu_der

class Sigmoid(Function):

    a = 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-(Sigmoid.a * x)))
    
    @staticmethod
    def sigmoid_der(x):
        return Sigmoid.sigmoid(x)*(1-Sigmoid.sigmoid(x))
    
    def activation(self):
        return Sigmoid.sigmoid

    def derivate(self):
        return Sigmoid.sigmoid_der

class Tanh(Function):

    a = 1

    @staticmethod
    def tanh(x):
        return np.tanh(Tanh.a * x / 2)

    @staticmethod
    def tanh_der(x):
        return Tanh.a * (1 - np.tanh(Tanh.a * x / 2) ** 2)/2

    def activation(self):
        return Tanh.tanh

    def derivate(self):
        return Tanh.tanh_der

class Type(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:

    def __init__(self, neurons, weights, activation_class, layer_type, weight_range = 0.7):
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

    def __init__(self, weight_range, hidden_layers_number, input_dimension, layer_length, activation_class_arr, seed = ""):
        self.depth = hidden_layers_number
        self.activation_class_arr = activation_class_arr
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
        self.seed = seed

    def set_reset(self):
        self.reset_hidden_layers = []
        for i in range(self.depth):
            self.reset_hidden_layers.append(np.copy(self.hidden_layers[i].weight_matrix))
        self.reset_output_layer = np.copy(self.output_layer.weight_matrix)

    def reset(self):
        for i in range(self.depth):
            self.hidden_layers[i].weight_matrix = np.copy(self.reset_hidden_layers[i])
        self.output_layer.weight_matrix = np.copy(self.reset_output_layer)

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

    def Loss_0_1(self, X, y, threshold, positive = 1, negative = 0):
        error = 0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            if output >= threshold:
                discrete_output = positive
            else:
                discrete_output = negative
            target_value = y.iloc[i]
            error += ((target_value - discrete_output))**2
        return error
    
    def Accuracy(self, X, y, threshold = 0.5):
        l = len(X)
        misclassified = self.Loss_0_1(X, y, threshold)
        return  ((l - misclassified) / l)

    def MSE(self, X, y):
        error = 0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            error += np.dot(y.iloc[i] - output, y.iloc[i] - output)
        return error / len(X)
    
    def MEE(self, X, y):
        error = 0
        for i in range(len(X)):
            output = self.network_output(X.iloc[i])
            error += np.sqrt(np.dot((y.iloc[i] - output), (y.iloc[i] - output)))
        return error / len(X)

    def backpropagation_batch(self, training_data, regression, mean, standardization, tollerance, max_epochs, eta, lambda_tichonov, alpha, other_data = None):
        X = training_data[0]
        y = training_data[1]
        l = len(X)
        if mean:
            eta = eta / l
        if standardization:
            self.std_mean["X_mean"] = X.mean()
            self.std_mean["X_std"] = X.std()
            self.std_mean["y_mean"] = y.mean()
            self.std_mean["y_std"] = y.std()
        X_std = (X - self.std_mean["X_mean"]) / self.std_mean["X_std"]
        y_std = (y - self.std_mean["y_mean"]) / self.std_mean["y_std"]
        error_function = self.MEE if regression else self.Accuracy
        task_errors = []
        MSE_errors = []
        task_other_errors = []
        MSE_other_errors = []
        batch_gradient = [[np.zeros((self.hidden_layers[i].neurons, self.hidden_layers[i].weights)) for i in range(self.depth)]for i in range(2)]
        batch_gradient[0].append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        batch_gradient[1].append(np.zeros((self.output_layer.neurons, self.output_layer.weights)))
        old_batch_gradient = [np.zeros((self.hidden_layers[i].neurons, self.hidden_layers[i].weights)) for i in range(self.depth)] ###
        old_batch_gradient.append(np.zeros((self.output_layer.neurons, self.output_layer.weights))) ###
        for i in range(max_epochs):
            task_errors.append(error_function(X, y))
            MSE_errors.append(self.MSE(X, y))
            if other_data:
                task_other_errors.append(error_function(other_data[0], other_data[1]))
                MSE_other_errors.append(self.MSE(other_data[0], other_data[1]))
            if len(MSE_errors) > 1:
                if MSE_errors[-1] > 10**4:
                    break
                if np.abs((MSE_errors[-2] - MSE_errors[-1])/ MSE_errors[-2]) * 100 < tollerance:
                    break
            for i in range(self.depth):
                batch_gradient[1][i] = lambda_tichonov * self.hidden_layers[i].weight_matrix
                self.hidden_layers[i].weight_matrix += alpha * old_batch_gradient[i]
            batch_gradient[1][-1] = lambda_tichonov * self.output_layer.weight_matrix
            self.output_layer.weight_matrix += alpha * old_batch_gradient[-1]
            batch_gradient[0] = self.backpropagation_iteration(X_std.iloc[0], y_std.iloc[0])
            for i in range(1, l):
                current_gradient = self.backpropagation_iteration(X_std.iloc[i], y_std.iloc[i])
                batch_gradient[0] = [batch_gradient[0][i] + current_gradient[i] for i in range(self.depth + 1)]
            for i in range(self.depth):
                batch_gradient[0][i] *= eta 
                self.hidden_layers[i].weight_matrix += (batch_gradient[0][i] - batch_gradient[1][i])
                old_batch_gradient[i] = alpha * old_batch_gradient[i] + batch_gradient[0][i]
            batch_gradient[0][-1] *= eta 
            self.output_layer.weight_matrix += (batch_gradient[0][-1] - batch_gradient[1][-1])
            old_batch_gradient[-1] = alpha * old_batch_gradient[-1] + batch_gradient[0][-1]

        return (MSE_errors, MSE_other_errors, task_errors, task_other_errors) if other_data else (MSE_errors, task_errors)
          
    def backpropagation_iteration(self, x, y):
        output = self.train_network_output(x)
        store_gradient = []
        current_layer_delta = np.zeros(self.output_layer.neurons)
        current_layer_der = self.output_layer.der_act(self.store_hidden_result[-1])
        updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[self.depth - 1]))
        current_layer_delta = (y - output) * current_layer_der
        store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
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
        next_layer_delta = current_layer_delta
        for hidden_layer_index in range(self.depth - 2, 0, -1):
            current_layer = self.hidden_layers[hidden_layer_index]
            current_layer_delta = np.zeros(current_layer.neurons)
            current_layer_der = current_layer.der_act(self.store_hidden_result[hidden_layer_index - 1])
            updated_hidden_result = np.concatenate((np.array([1]), self.store_hidden_result[hidden_layer_index - 1]))
            for index_neuron in range(current_layer.neurons):
                current_layer_delta[index_neuron] = np.dot(next_layer_delta, self.hidden_layers[hidden_layer_index + 1].weight_matrix[:,index_neuron + 1]) * current_layer_der[index_neuron]
            store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
            next_layer_delta = current_layer_delta
        current_layer = self.hidden_layers[0]
        current_layer_delta = np.zeros(current_layer.neurons)
        current_layer_der = current_layer.der_act(x)
        updated_hidden_result = np.concatenate((np.array([1]), x))
        for index_neuron in range(current_layer.neurons):
            current_layer_delta[index_neuron] = np.dot(next_layer_delta, self.hidden_layers[1].weight_matrix[:,index_neuron + 1]) * current_layer_der[index_neuron]
        store_gradient.append(np.outer(current_layer_delta, updated_hidden_result))
        store_gradient.reverse()
        return store_gradient
    
    def internal_grid_search(self, training_data, validation_data, regression, mean, standardization, tollerance, max_epochs, eta_range, lambda_tichonov_range, alpha_range, save):
        best_comb = -1
        best_model = [0, 0, 0]
        best_t_l = []
        best_t_t = []
        best_v_l = []
        best_v_t = [np.inf]
        counter = 0
        self.set_reset()
        for current_eta in eta_range:
            for current_lambda_tichonov in lambda_tichonov_range:
                for current_alpha in alpha_range:
                    loss_t_error, loss_v_error, task_t_error, task_v_error = self.backpropagation_batch(training_data, regression, mean, standardization, tollerance, max_epochs, current_eta, current_lambda_tichonov, current_alpha, validation_data)
                    if save:
                        plot_error(loss_t_error, loss_v_error, "Validation", "Validation", "Seed_" + str(self.seed) + "_" + str(self.hidden_layers[0].neurons) + "_" + str(current_eta) + "_" + str(current_lambda_tichonov) + "_" + str(current_alpha) + "_Validation_MSE", save, False)
                        plot_error(task_t_error, task_v_error, "Validation", "Validation", "Seed_" + str(self.seed) + "_" + str(self.hidden_layers[0].neurons) + "_" + str(current_eta) + "_" + str(current_lambda_tichonov) + "_" + str(current_alpha) + "_Validation_MEE", save, False)
                        print(f"({self.seed}) Validation error: {task_v_error[-1]}, Training error: {task_t_error[-1]}, net: {[self.hidden_layers[i].neurons for i in range(self.depth)]}, params: {current_eta, current_lambda_tichonov, current_alpha}")
                    if task_v_error[-1] < best_v_t[-1]:
                        best_comb = counter
                        best_model = [current_eta, current_lambda_tichonov, current_alpha]
                        best_t_l = loss_t_error
                        best_t_t = task_t_error
                        best_v_l = loss_v_error
                        best_v_t = task_v_error
                    counter += 1
                    self.reset()
        plot_error(best_t_l, best_v_l, "Validation", "Validation", "Seed_" + str(self.seed) + "_Validation_MSE", True, False)
        plot_error(best_t_t, best_v_t, "Validation", "Validation", "Seed_" + str(self.seed) + "_Validation_MEE", True, False)
        return [best_comb, best_model, np.round(best_v_t[-1], 4), np.round(best_t_t[-1], 4)]

    def save_net(self, filename):
        with open("Weights/" + filename + ".txt", "w") as f:
            for i, layer in enumerate(self.hidden_layers):
                f.write(f"Layer {i}:\n")
                np.savetxt(f, layer.weight_matrix, fmt='%.6f')
                f.write("\n")
            f.write("Output Layer:\n")
            np.savetxt(f, self.output_layer.weight_matrix, fmt='%.6f')
            f.write(f"\nStandardization:\n")
            np.savetxt(f, [self.std_mean["X_mean"]], fmt='%.6f')
            np.savetxt(f, [self.std_mean["X_std"]], fmt='%.6f')
            np.savetxt(f, [self.std_mean["y_mean"]], fmt='%.6f')
            np.savetxt(f, [self.std_mean["y_std"]], fmt='%.6f')
        f.close()

    def load_weights(self, filename):
        with open("Weights/" + filename + ".txt", 'r') as f:
            lines = f.readlines()
        
        lines = iter(lines)

        current_layer_weights = []
        current_layer = self.hidden_layers[0]
        for i, line in enumerate(lines):
            if line.startswith("Layer"):
                layer_number = int(line.split()[1].strip(':'))
                current_layer = self.hidden_layers[layer_number]
            elif line.startswith("Output"):
                current_layer = self.output_layer
            elif line.startswith("Standardization"):
                line = next(lines)
                self.std_mean["X_mean"] = [float(x) for x in line.split()]
                line = next(lines)
                self.std_mean["X_std"] = [float(x) for x in line.split()]
                line = next(lines)
                self.std_mean["y_mean"] = [float(x) for x in line.split()]
                line = next(lines)
                self.std_mean["y_std"] = [float(x) for x in line.split()]
            elif not line.strip():
                current_layer.weight_matrix = np.array(current_layer_weights)
                current_layer_weights = []
            else:
                current_layer_weights.append([float(x) for x in line.split()])
