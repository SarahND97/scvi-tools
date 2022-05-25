from scvi import model
from scvi.data._anndata import _setup_anndata
import numpy as np
from scvi._settings import settings
from experiment import *


def create_imputation_error(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # select non-zero subset
    (i,j) = np.nonzero(X_zero)
    
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
       
    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix

def imputation_error(X_mean, X, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x - y))

# Imputation on B Cell data: 
gene_indexes, data_cross, adata_model = data_bcell(5)
test_data = data_cross[0]
data_cross = concatenate_adatas(data_cross[1:])
data = concatenate_adatas([data_cross, adata_model])
corrupted_data, row, column, indexes = create_imputation_error(test_data)
# print(all_data) # 0.0003, 2, 512
_setup_anndata(data, labels_key="labels")
model_ = model.HYBRIDVI(data, gene_indexes, 512, n_layers=2)
model_.train(lr=0.0003)
expression = model_.get_normalized_expression(corrupted_data, n_samples=10, return_mean=True)
error = imputation_error(expression, test_data, row, column, indexes)
print(error)
f = open("wilcoxon_results/imputation_bcell.txt","a")
f.write("Hybrid imputation error bcell: " + str(error) + "\n")
f.close()

settings.seed=0
_, data_cross, adata_model = data_bcell(5)
test_data = data_cross[0]
data_cross = concatenate_adatas(data_cross[1:])
data = concatenate_adatas([data_cross, adata_model])
corrupted_data, row, column, indexes = create_imputation_error(test_data)
# print(all_data) # 0.0003, 2, 512
_setup_anndata(data, labels_key="labels")
model_ = model.SCVI(data, n_hidden=128, n_layers=1)
model_.train(lr=0.0004)
expression = model_.get_normalized_expression(corrupted_data, n_samples=10, return_mean=True)
error = imputation_error(expression, test_data, row, column, indexes)
print(error)
f = open("wilcoxon_results/imputation_bcell.txt","a")
f.write("scVI imputation error bcell: " + str(error) + "\n")
f.close()

