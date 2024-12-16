import pandas as pd
from NN.Neural_Network import *

cup_data = pd.read_csv("Dataset/Cup/ML-CUP24-TR.csv", sep=",", header=None, comment="#")

n_colonne = cup_data.shape[1]
colonne = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)] + ['target_x', 'target_y', 'target_z']
cup_data.columns = colonne

# Separazione input e target
X = cup_data.drop(columns=["datanumber", 'target_x', 'target_y', 'target_z'])
y = cup_data[["target_x", "target_y", "target_z"]]

X_stand = (X - X.mean()) / X.std()
y_stand = (y - y.mean()) / y.std()

train_size = int(0.33 * len(X_stand))
X_train = X_stand.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X_stand.iloc[train_size:]
y_test = y.iloc[train_size:]

network = Network(0.5, 3, X.shape[1], [5, 5, 5, 3], [Tanh(0.1), Tanh(0.1), Tanh(0.1), Id()])

#network.plot_target(y, "")

network.backpropagation_batch(X_train, y_train, batches_number = 300, eta = 0.025, lambda_tichonov = 0.05, alpha = 0, validation = [X_test, y_test], plot = True)

print(f"{network.LMS_regression(X_train, y_train, True)}")

print(f"{network.LMS_regression(X_test, y_test, True)}")

print(f"{network.LED_regression(X_train, y_train, True)}")

print(f"{network.LED_regression(X_test, y_test, True)}")

network.plot_output(X, "")

network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1])