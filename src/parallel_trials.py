from trial import *
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import time

def compute_mean_var(seeds):

    tr_results, val_results, ts_results = [], [], []

    with open("trials.txt", "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("seed:"):  
            try:
                seed = int(line.split()[1])
                if seed in seeds:
                    training_error = float(line.split("training_error:")[1].split()[0])
                    validation_error = float(line.split("validation_error:")[1].split()[0])
                    test_error = float(line.split("test_error:")[1].split()[0])
                    tr_results.append(training_error)
                    val_results.append(validation_error)
                    ts_results.append(test_error)
                    seeds.remove(seed)
            except Exception as e:
                print(f"Errore nell'estrazione degli errori: {e}")

    tr_mean, tr_var = np.mean(tr_results), np.var(tr_results)
    val_mean, val_var = np.mean(val_results), np.var(val_results)
    ts_mean, ts_var = np.mean(ts_results), np.var(ts_results)

    print(f"Training - Media: {tr_mean:.4f}, Varianza: {tr_var:.4f}")
    print(f"Validation - Media: {val_mean:.4f}, Varianza: {val_var:.4f}")
    print(f"Test - Media: {ts_mean:.4f}, Varianza: {ts_var:.4f}")


def multiple_trials(seeds):

    start = time.time()

    with ProcessPoolExecutor() as executor:

        futures = {executor.submit(trial, seed, False): seed for seed in seeds}
        
        wait(futures)

        results = []

        for future in as_completed(futures):
            try:
                result, network, params = future.result()
                seed = futures[future]  
                
                results.append({
                    "seed": seed,
                    "training_error": result["training_error"],
                    "validation_error": result["validation_error"],
                    "test_error": result["test_error"],
                    "network": [layer.neurons for layer in network.hidden_layers] + [network.output_layer.neurons],
                    "activation_class": [type(x).__name__ for x in network.activation_class_arr],
                    "params": params,
                })
            
            except Exception as e:
                print(f"Error in future: {e}")

    results_sorted = sorted(results, key=lambda x: x["seed"])

    with open("trials.txt", "a") as f:
        for res in results_sorted:
            f.write(
                f"\n\nseed: {res['seed']}\ttraining_error: {res['training_error']}\t"
                f"validation_error: {res['validation_error']}\ttest_error: {res['test_error']}\n"
            )
            f.write(
                f"network: {res['network']}\tactivation_function: {res['activation_class']}\t"
                f"params: {res['params']}\n"
            )

    end = time.time()
    print(f"({seeds}) Total elapsed time: {round((end - start) / 60, 2)} minutes")

if __name__ == "__main__":
    # trial used for computing mean and std of training, validation and test error
    # seeds = [1, 2, 3, 4, 5, 6]
    seeds = [4, 5, 6]
    multiple_trials(seeds)
    # compute_mean_var(seeds)
