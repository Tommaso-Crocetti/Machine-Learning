import pandas as pd
from NN.Neural_Network import *
#----------------------------------------------------------------------------------------------------------------------------
cup_data = pd.read_csv("Dataset/Cup/ML-CUP24-TR.csv", sep=",", header=None, comment="#")
cup_data_TS = pd.read_csv("Dataset/Cup/ML-CUP24-TS.csv", sep=",", header=None, comment="#")

n_colonne = cup_data.shape[1]
colonne = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)] + ['target_x', 'target_y', 'target_z']
colonne_TS = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)]

cup_data.columns = colonne
cup_data_TS.columns = colonne_TS

X_TS = cup_data_TS.drop(columns=["datanumber"])
X = cup_data.drop(columns=["datanumber", 'target_x', 'target_y', 'target_z'])
y = cup_data[["target_x", "target_y", "target_z"]]
#----------------------------------------------------------------------------------------------------------------------------
train_size = int(0.66 * len(X))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

TR_stand = {
    "X_train_std": X_train.std(),
    "y_train_std": y_train.std(),
    "X_train_mean": X_train.mean(),
    "y_train_mean": y_train.mean()
}

#----------------------------------------------------------------------------------------------------------------------------
network = Network(0.75, 3, X.shape[1], [15, 20, 15, 3], [ Tanh(0.1), Tanh(0.1),Tanh(0.1), Id()], TR_stand)

network.backpropagation_batch(X_train, y_train, batches_number = 100, eta = 0.05, lambda_tichonov = 0.005, alpha = 0.025, validation = [X_test, y_test], plot = True)

print(network.LED_regression(X_test, y_test, True))

#network.K_fold_CV(X_train, y_train, batches_number = 100, eta = 0.1, lambda_tichonov = 0.005, alpha = 0.025, plot = False, fold_number = 4)
#----------------------------------------------------------------------------------------------------------------------------

'''

network.plot_target(y_stand, "")

network.plot()

for i in range(len(X)):
    print(network.network_output((X.iloc[i]) * y.std()) + y.mean())

print(f"{network.LMS_regression(X_train, y_train, True)}")

print(f"{network.LMS_regression(X_test, y_test, True)}")

print(f"{network.LED_regression(X_train, y_train, True)}")

print(f"{network.LED_regression(X_test, y_test, True)}")


'''