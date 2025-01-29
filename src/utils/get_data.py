import pandas as pd

def get_data_monk(id, encoding):
    data = pd.read_csv("Dataset/Monk/monks-" + str(id) + ".train",sep = "\s+", header=None) 
    test_data = pd.read_csv("Dataset/Monk/monks-" + str(id) + ".test", sep = "\s+", header=None)

    n_colonne = data.shape[1]
    colonne = ['target'] + [ f'feature{i}' for i in range(1,n_colonne -1)] + ['datanumber']
    
    data.columns = colonne
    test_data.columns = colonne

    X = data.drop(columns=["target", 'datanumber'])
    y = data["target"]

    test_X = test_data.drop(columns=["target", 'datanumber'])
    test_y = test_data["target"]

    if encoding:
        X = pd.get_dummies(X, dtype=float, columns=[f'feature{i}' for i in range(1, n_colonne - 1)])
        test_X = pd.get_dummies(test_X, dtype=float, columns=[f"feature{i}" for i in range(1, n_colonne - 1)])
    return [X, y], [test_X, test_y]

def hold_out_cup(train_perc, validation_perc):
    data = pd.read_csv("Dataset/Cup/ML-CUP24-TR.csv", sep=",", header=None, comment="#")
    n_colonne = data.shape[1]
    colonne = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne - 3)] + ['target_x', 'target_y', 'target_z']
    data.columns = colonne
    X = data.drop(columns=["datanumber", 'target_x', 'target_y', 'target_z']) 
    y = data[["target_x", "target_y", "target_z"]]
    l = len(X)
    train_size = int(train_perc * l)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    validation_size = int(validation_perc * l)
    X_val = X.iloc[train_size:train_size+validation_size]
    y_val = y.iloc[train_size:train_size+validation_size]
    X_test = X.iloc[train_size + validation_size:]
    y_test = y.iloc[train_size + validation_size:]
    return [X_train, y_train], [X_val, y_val], [X_test, y_test]

def get_blind_TS():
    data = pd.read_csv("Dataset/Cup/ML-CUP24-TS.csv", sep=",", header=None, comment="#")
    n_colonne = data.shape[1]
    colonne = ['datanumber'] + [f'feature{i}' for i in range(1, n_colonne)]
    data.columns = colonne
    X = data.drop(columns=["datanumber"])
    return X