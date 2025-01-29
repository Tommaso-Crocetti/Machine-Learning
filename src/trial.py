from utils.get_data import *
from utils.grid_search import *

def trial(seed, show = False):

    np.random.seed(seed)

    result = {"training_error": None, "validation_error": None, "test_error": None}
    
    training_data, validation_data, test_data = hold_out_cup(0.5, 0.25)

    network0_20 = Network(0.7, 2, training_data[0].shape[1], [20, 20, 3], [Tanh(), Tanh(), Id()], seed)

    network0_35 = Network(0.7, 2, training_data[0].shape[1], [35, 35, 3], [Tanh(), Tanh(), Id()], seed)

    network0_50 = Network(0.7, 2, training_data[0].shape[1], [50, 50, 3], [Tanh(), Tanh(), Id()], seed)

    start = time.time()

    network, params, validation_error = grid_search([network0_20, network0_35, network0_50], training_data, validation_data, True, True, True, 0.01, 500, [0.02, 0.05, 0.08], [10**-2, 10**-3, 10**-4], [0.7, 0.55, 0.4], ["20", "35", "50"])

    network.set_reset()

    end = time.time()

    result["validation_error"] = validation_error

    print(f"({seed}) Validation error: {validation_error}, net: {[network.hidden_layers[i].neurons for i in range(network.depth)]}, {[type(network.activation_class_arr[i]).__name__ for i in range(network.depth)]} params: {params}")

    training_data = [pd.concat((training_data[0], validation_data[0])), pd.concat((training_data[1], validation_data[1]))]
    
    network.reset()

    start = time.time()

    training_error, test_error, task_training_error, task_test_error = network.backpropagation_batch(training_data, True, True, True, 0.01, 1000, params[0], params[1], params[2], test_data)
    
    end = time.time()

    network.save_net(str(seed))

    print(f"({seed}) Test error: {task_test_error[-1]}, test elapsed time: {(end - start) // 60} minutes")

    plot_error(training_error, test_error, "Test", "Test", "Seed_" + str(seed) + "_Test_MSE", True, show)
    plot_error(task_training_error, task_test_error, "Test", "Test", "Seed_" + str(seed) + "_Test_MEE", True, show)
    
    result["test_error"] = round(task_test_error[-1], 4)

    network.reset()

    training_data = [pd.concat((training_data[0], test_data[0])), pd.concat((training_data[1], test_data[1]))]

    start = time.time()

    training_error, task_training_error = network.backpropagation_batch(training_data, True, True, True, 0.01, 1000, params[0], params[1], params[2])

    end = time.time()

    network.save_net(str(seed) + "_retrained")

    print(f"({seed}) Training error: {task_training_error[-1]}, train elapsed time: {(end - start) // 60} minutes")

    plot_error(training_error, None, None, "Train", "Seed_" + str(seed) + "_Train_MSE", True, show)
    plot_error(task_training_error, None, None, "Train", "Seed_" + str(seed) + "_Train_MEE", True, show)

    result["training_error"] = round(training_error[-1], 4)

    end = time.time()
    
    print(f"({seed}) Total elapsed time: {round((end - start) / 60, 2)} minutes")

    return result, network, params

if __name__ == "__main__":
    trial(4, True)