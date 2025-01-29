from utils.get_data import *
from utils.Neural_Network import *
import time

def monk_trial(id, seed, show = True):

    training_data, test_data = get_data_monk(id, True)

    np.random.seed(seed)

    network = Network(0.1, 1, 17, [4, 1], [Relu(), Sigmoid()], seed)

    start = time.time()

    training_error, test_error, training_task_error, test_task_error = network.backpropagation_batch(training_data, False, True, False, 0, 500, 0.5, 0, 0.9, test_data)

    end = time.time()

    print(f"({seed}) Training MSE error: {training_error[-1]}, Test MSE error: {test_error[-1]}")

    print(f"({seed}) Training accuracy: {training_task_error[-1]}, Test accuracy: {test_task_error[-1]}")

    print(f"({seed}) Elapsed time: {(end - start)}")

    plot_error(training_error, test_error, "Test", "Monk" + str(id), "Seed_" + str(seed) + "_Test_MSE", True, show)
    plot_error(training_task_error, test_task_error, "Test", "Monk" + str(id), "Seed_" + str(seed) + "_Test_MEE", True, show)


monk_trial(1, 6, False)