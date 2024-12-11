import pandas as pd
from NN.Neural_Network import *

cup_data = pd.read_csv("Dataset/Cup/ML-CUP24-TR.csv", sep=",", header=None, comment="#")

n_colonne = cup_data.shape[1]
colonne = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)] + ['target_x', 'target_y', 'target_z']
cup_data.columns = colonne

# Separazione input e target
X = cup_data.drop(columns=["datanumber", 'target_x', 'target_y', 'target_z'])
y = cup_data[["target_x", "target_y", "target_z"]]

network = Network(0.25, 3, X.shape[1], [3,3,3,3], [Relu(), Relu(), Relu(), Id()])

#network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

network.backpropagation_batch(X, y, batches_number = 200, eta = 0.01, lambda_tikonov= 0.001, alpha = 0.01, plot = True)

print(network.LMS_classification(X, y))

#network.plot_from([1,1,1,1,1,1,1,1,1,1,1,1])