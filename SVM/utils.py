import numpy as np

def k_fold_data(k : int, X: np.array):
    start_index = 0
    end_index = k
    k_folds = []

    for i in range(k):
        k_folds.append(X[start_index:end_index])
        start_index = end_index
        end_index += k
    
    return k_folds


