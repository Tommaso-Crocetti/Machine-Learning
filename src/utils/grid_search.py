from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import time
from utils.Neural_Network import *

def write_result(network, net_name, best_comb, best_model, best_validation_error, training_error):
    with open("Grid_search/" + net_name + "_grid_search.txt", "a") as f:
        f.write(f"\n|\t{best_comb}\t\t|\t{best_model}\t|\t{best_validation_error}\t|\t{training_error}\t"
                f"|\t{network.seed}\t|\t{[network.hidden_layers[i].neurons for i in range(network.depth)] + [network.output_layer.neurons]}\t"
                f"|\t{list(map(lambda x: type(x).__name__, network.activation_class_arr))}\t|\n")
    f.close()

def grid_search_iteration(networks, training_data, validation_data, regression, mean, standardization, tollerance, max_epochs, eta_range, lambda_range, alpha_range, prefixes):
    start = time.time()
    best_result = None 
    network_index = None

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(network.internal_grid_search, training_data, validation_data, regression, mean, standardization, tollerance, max_epochs, eta_range, lambda_range, alpha_range, prefix, False): i
            for i, (network, prefix) in enumerate(zip(networks, prefixes))
        }

        wait(futures)

        results = []

        for future in as_completed(futures):
            try:
                result = future.result()
                index = futures[future] 
                
                results.append((index, result))

                if best_result is None or result[2] < best_result[2]: 
                    best_result = result
                    network_index = index

            except Exception as e:
                print(f"Error in future for network {futures[future]}: {e}") 

    results.sort(key=lambda x: x[0])

    for index, result in results:
        write_result(networks[index], prefixes[index], result[0], result[1], result[2], result[3])
        print(f"({networks[index].seed}) Result for network {prefixes[index]}: (val) {result[2]}, (train) {result[3]} params: {result[1]} (combination {result[0]})")  # Debug

    end = time.time()
    print(f"Single grid elapsed time: {round((end - start) / 60, 2)} minutes")

    return network_index, best_result[1], best_result[2]

def grid_search(networks, training_data, validation_data, regression, mean, standardization, tollerance, max_epochs, eta_range, lambda_range, alpha_range, prefixes):

    start = time.time()
    
    coarse_network_index, coarse_params, _ = grid_search_iteration(networks, training_data, validation_data, regression, mean, standardization, tollerance, max_epochs, eta_range, lambda_range, alpha_range, prefixes)

    seed = networks[coarse_network_index].seed

    refined_unit_1 = [networks[coarse_network_index].hidden_layers[i].neurons - 5 for i in range(networks[coarse_network_index].depth)]
    refined_unit_1.append(networks[coarse_network_index].output_layer.neurons)

    refined_unit_2 = [networks[coarse_network_index].hidden_layers[i].neurons + 5 for i in range(networks[coarse_network_index].depth)]
    refined_unit_2.append(networks[coarse_network_index].output_layer.neurons)

    refined_net_1 = Network(0.7, networks[coarse_network_index].depth, training_data[0].shape[1], refined_unit_1, networks[coarse_network_index].activation_class_arr, networks[coarse_network_index].seed)
    refined_net_2 = Network(0.7, networks[coarse_network_index].depth, training_data[0].shape[1],refined_unit_2, networks[coarse_network_index].activation_class_arr, networks[coarse_network_index].seed)

    networks = [refined_net_2, networks[coarse_network_index], refined_net_1]

    refined_eta_range = [round(coarse_params[0] + 0.01, 3), coarse_params[0], round(coarse_params[0] - 0.01, 3)]
    refined_lambda_range = [round(coarse_params[1] + coarse_params[1] / 2, 6), coarse_params[1], round(coarse_params[1] - coarse_params[1] / 2, 6)]
    refined_alpha_range = [round(coarse_params[2] + 0.05, 2), coarse_params[2], round(coarse_params[2] - 0.05, 2)]
    refined_prefixes = ["refined_0", "refined_1", "refined_2"]

    refined_network_index, refined_params, validation_error = grid_search_iteration(networks, training_data, validation_data, regression, mean, standardization, tollerance, max_epochs, refined_eta_range, refined_lambda_range, refined_alpha_range, refined_prefixes)

    end = time.time()

    print(f"({seed}) Total grid elapsed time: {round((end - start) / 60, 2)} minutes")

    return networks[refined_network_index], refined_params, validation_error