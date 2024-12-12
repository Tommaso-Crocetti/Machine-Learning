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


network = Network(0.25, 5, X.shape[1], [3,5,6,5,4,3], [Tanh(10),Tanh(10),Tanh(10),Tanh(10),Relu(),Id()])

network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1])

network.backpropagation_batch(X_stand, y, batches_number = 100, eta = 0.001, lambda_tikonov= 0.005, alpha = 0.001, plot = True)

print(f"{round(network.LMS_regression(X_stand, y, True)/2.5, 2)}%")

network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1])