from utils.get_data import *
from utils.Neural_Network import *

training_data, validation_data, test_data = hold_out_cup(0.5, 0.25)

np.random.seed(3)

network0_20 = Network(0.7, 2, training_data[0].shape[1], [20, 20, 3], [Tanh(), Tanh(), Id()], 4)

network0_35 = Network(0.7, 2, training_data[0].shape[1], [35, 35, 3], [Tanh(), Tanh(), Id()], 4)

network0_50 = Network(0.7, 2, training_data[0].shape[1], [50, 50, 3], [Tanh(), Tanh(), Id()], 4)

training_error, test_error, task_training_error, task_test_error = network0_50.backpropagation_batch(training_data, True, True, False, 0.01, 10, 0.09, 0.001, 0.75, test_data)

print(training_error[-1], test_error[-1], task_training_error[-1], task_test_error[-1])

plot_output(network0_35, training_data[0], training_data[1], "", False, True)
plot_output(network0_35, test_data[0], test_data[1], "", False, True)

print(plot_error(training_error, test_error, "Test", "Test", "Test_MSE", True, True))
print(plot_error(task_training_error, task_test_error, "Test", "Test", "Test_MEE", True, True))