from scvi import model
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from scvi.data._built_in_data._pbmc import _load_pbmc_dataset
from scvi.data._anndata import _setup_anndata
import scanpy as sc
import sys
import numpy as np

def clustering_scores(labels_true, labels_pred):
    return [NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]

# Find parameters 
# the parameters are learning rate, layers, size of layers
def cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data, K):
    results = []
    average_nmi = 0
    average_ari = 0
    parameters = [learning_rate, hidden_layers, size_hidden_layer]
    # might need to change the name of the text file every time 
    f = open("output/average_results.txt","a")
    f2 = open("output/results.txt","a")
    for i in range(K):
        data_ = list(data)
        test_i = data_[i]
        train_i = data_.pop(i)
        model_ = model.HYBRIDVI(adata=train_i, gene_indexes=gene_indexes_von_mises, n_hidden=size_hidden_layer, n_layers=hidden_layers)
        model_.train(lr=learning_rate)
        # TODO: create a get_latent specific for hybridVI
        latent = model_.get_latent_representation(adata=test_i, hybrid=True)
        test_i.obsm["X_scvi"] = latent
        sc.pp.neighbors(test_i, n_neighbors=10, n_pcs=40, use_rep="X_scvi")
        sc.tl.leiden(test_i, key_added="leiden_scvi", resolution=0.5)
        pred = test_i.obs["leiden_scvi"].to_list()
        pred = [int(x) for x in pred]
        scores_leiden = clustering_scores(test_i.obs["labels"], pred)
        result = [scores_leiden[0], scores_leiden[1]]
        results.append(result)
        average_nmi = average_nmi + scores_leiden[0] 
        average_ari = average_ari + scores_leiden[1]

    # Calculate the average over K-folds 
    average = [average_nmi/(K), average_ari/(K)]
    # add results to files
    f.write(str(parameters) + " " + str(average) + " \n")
    f2.write(str(parameters) + " " + str(results) + " \n")
    f.close()
    f2.close()
    print(average)

adata = _load_pbmc_dataset(run_setup_anndata=False)
K_cross = 3

# Find the cell cycle genes in the data
cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
genes = [x for x in adata.var["gene_symbols"]]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
adata.var["von_mises"] = "false"
for gene in cell_cycle_genes:
    adata.var.loc[adata.var["gene_symbols"] == gene, "von_mises"] = "true"
gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]


adata_cross = adata[:int((len(adata)*2)/K_cross), :]
adata_train_best_model = adata[int((len(adata)*2)/K_cross):, :]
size = int(len(adata_cross)/K_cross)
adata_1 = adata_cross[:size,:]
adata_2 = adata_cross[size:2*size,:]
adata_3 = adata_cross[2*size:,:]
data = [adata_train_best_model, adata_1, adata_2, adata_3]
for d in range(len(data)):
    da = data[d].copy()
    _setup_anndata(da, batch_key="batch", labels_key="labels")
    data[d] = da 

data_cross = [data[1], data[2], data[3]]
file = "input/parameters.in"
line_nr = int(sys.argv[1])
f = open(file, "r")
lines = f.readlines()
line = lines[line_nr].split()
print(line)
learning_rate = float(line[0])
hidden_layers = int(line[1])
size_hidden_layer = int(line[2])
cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data_cross, K_cross)
f.close()