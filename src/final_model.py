from utils.get_data import *
from utils.Neural_Network import *

def find_final_model(seed):

    with open("trials.txt", "r") as f:
        lines = f.readlines()

    lines = iter(lines)

    network = None
    activation_functions = []
    params = []

    for line in lines:
        if line.startswith("seed:"):  
            current_seed = int(line.split()[1])
            if current_seed == seed:
                next_line = next(lines)
                network_str = next_line.split("network:")[1].split("activation_function:")[0].strip()
                network = eval(network_str)
                
                activation_function_str = next_line.split("activation_function:")[1].split("params:")[0].strip()
                activation_functions = [eval(x)() for x in eval(activation_function_str)]

                break

    network = Network(0.7, len(network) - 1, 12, network, activation_functions, seed)
    network.load_weights(str(seed) + "_retrained")
    return network

def compute_result(seed):
    X = get_blind_TS()
    network = find_final_model(seed)
    with open("Newtork_ML-CUP24-TS.csv", "w") as f:
        f.write("# Tommaso Crocetti, Pietro Lorenzo Bianchi\n")
        f.write("# Newtork\n")
        f.write("# ML-CUP24 v1\n")
        f.write("# 07/01/2025\n")
        for i in range(len(X)):
            f.write(f"{i + 1}," + ",".join(map(str, network.network_output(X.iloc[i]))) + "\n")

if __name__ == "__main__":
    # randomly chosen seed for blind test set
    seed = 6
    compute_result(6)