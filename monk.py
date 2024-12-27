import pandas as pd
from NN.Neural_Network import *

monk_s_problems = pd.read_csv("Dataset/Monk/monks-3.train",sep = "\s+", header=None) 
test_set = pd.read_csv("Dataset/Monk/monks-3.test", sep = "\s+", header=None)

# data (as pandas dataframes) 
n_colonne = monk_s_problems.shape[1]
colonne = ['target'] + [ f'feature{i}' for i in range(1,n_colonne -1)] + ['datanumber']
monk_s_problems.columns = colonne
test_set.columns = colonne

X = monk_s_problems.drop(columns=["target", 'datanumber'])
y = monk_s_problems["target"]

test_X = test_set.drop(columns=["target", 'datanumber'])
test_y = test_set["target"]


X_encoded = pd.get_dummies(X, dtype=float, columns=[f'feature{i}' for i in range(1, n_colonne - 1)])
test_X_encoded = pd.get_dummies(test_X, dtype=float, columns=[f"feature{i}" for i in range(1, n_colonne - 1)])

network = Network(0.5, 2, X_encoded.shape[1], [3,3,1], [Sigmoid(1),Sigmoid(1),Sigmoid(1)])

network.backpropagation_batch(X_encoded, y, regression = False, mean = False, max_batches_number = 200, eta = 0.1, lambda_tichonov = 0.001, alpha = 0,validation = [test_X_encoded, test_y], plot = True)

network.plot()
