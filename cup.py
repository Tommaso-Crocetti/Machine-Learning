import pandas as pd
from NN.Neural_Network import *
#----------------------------------------------------------------------------------------------------------------------------
cup_data = pd.read_csv("Dataset/Cup/ML-CUP24-TR.csv", sep=",", header=None, comment="#")
cup_data = cup_data.sample(frac=1).reset_index(drop=True)

n_colonne = cup_data.shape[1]
colonne = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)] + ['target_x', 'target_y', 'target_z']
colonne_TS = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)]

cup_data.columns = colonne
cup_data_TS.columns = colonne_TS

# Separazione input e target
X = cup_data.drop(columns=["datanumber", 'target_x', 'target_y', 'target_z']) 
y = cup_data[["target_x", "target_y", "target_z"]]

train_size = int(0.66 * len(X))
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

network1 = Network(0.5, 3, X.shape[1], [15, 20, 15, 3], [Relu(), Relu(), Relu(), Id()])

network2 = Network(0.5, 4, X.shape[1], [15, 20, 20, 15, 3], [Relu(), Relu(), Relu(), Relu(), Id()])

network3 = Network(0.5, 5, X.shape[1], [15, 20, 30, 20, 15, 3], [Relu(), Relu(), Relu(), Relu(), Id()])

print(network1.grid_search([X_train, y_train], [X_test, y_test], [-4, -3, -2], [-4, -3, -2], [-4, -3, -2], "net1"))

'''
print(grid_search([X_train, y_train], [X_test, y_test], Relu()))

network.backpropagation_batch(X_train, y_train, standardization = True, tollerance = 3, max_batches_number = 1000, eta = 0.005, lambda_tichonov = 0.0005, alpha = 0.0025, validation = [X_test, y_test], plot = True)

network.plot_output(X, y, "Cup")

network.plot("Cup")

print(f"{network.LMS_regression(X_train, y_train, True)}")

print(f"{network.LMS_regression(X_test, y_test, True)}")

print(f"{network.LED_regression(X_train, y_train, True)}")

print(f"{network.LED_regression(X_test, y_test, True)}")

'''