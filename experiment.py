from scvi import model
import anndata
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from scvi.data._built_in_data._pbmc import _load_pbmc_dataset
from scvi.data._built_in_data._cortex import _load_cortex
from scvi.data._anndata import _setup_anndata
import scanpy as sc
import numpy as np
from scipy.stats import wilcoxon 
import muon as mu
from sklearn import preprocessing
from scvi._settings import settings
from sklearn.cluster import KMeans
import copy


def concatenate_adatas(list_adata):
    return anndata.AnnData.concatenate(*list_adata,batch_key='batch')

def create_parameters_file():
    learning_rate = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
    hidden_layer_von_mises = [1,2]
    size_hidden_layer_von_mises = [64,128,256,512]
    i = 0
    name = "input/parameters.in"
    for l in learning_rate:
        for h_l in hidden_layer_von_mises:
            for s_h_l in size_hidden_layer_von_mises:
                    f = open(name,"a")
                    string = str(l) + " " + str(h_l) + " " + str(s_h_l) + "\n"
                    f.write(string)
                    f.close
                    i = i+1
                    print(i)

def clustering_scores(labels_true, labels_pred):
    return [NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]

def divide_data_without_setup(data, K):
    divided_data = [[] for _ in range(K)]
    # make sure that there is an equal amount of labels in each dataset
    for i in range(K):
        for label in set(data.obs["labels"]):
            label_list = data[np.where(data.obs["labels"] == label)[0]]
            total = len(label_list)
            size = int(total/K)
            divided_data[i].append(label_list[i*size:size*(i+1),:])

    return [concatenate_adatas(d) for d in divided_data]

def divide_data(data, K):
    divided_data = [[] for _ in range(K)]
    # make sure that there is an equal amount of labels in each dataset
    for i in range(K):
        for label in set(data.obs["labels"]):
            label_list = data[np.where(data.obs["labels"] == label)[0]]
            total = len(label_list)
            size = int(total/K)
            divided_data[i].append(label_list[i*size:size*(i+1),:])

    divided_data = [concatenate_adatas(d) for d in divided_data]
    for i in range(len(divided_data)):
        da = divided_data[i].copy()
        _setup_anndata(da, labels_key="labels")
        divided_data[i] = da 
    return divided_data

# Find parameters 
# the parameters are learning rate, layers, size of layers
def cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data, K, filename, res):
    results = []
    average_nmi = 0
    average_ari = 0
    parameters = [learning_rate, hidden_layers, size_hidden_layer]
    f = open("cross_valid_results/" + filename + "average_results.txt","a")
    f2 = open("cross_valid_results/" + filename + "results.txt","a")
    for i in range(K):
        settings.seed=0
        train_i = copy.deepcopy(data)
        test_i = train_i[i]
        train_i.pop(i)
        train_i = concatenate_adatas(train_i)
        _setup_anndata(test_i, labels_key="labels")
        _setup_anndata(train_i, labels_key="labels")
        model_ = model.HYBRIDVI(adata=train_i, gene_indexes=gene_indexes_von_mises, n_hidden=size_hidden_layer, n_layers=hidden_layers)
        model_.train(lr=learning_rate)
        latent = model_.get_latent_representation(adata=test_i, hybrid=True)
        test_i.obsm["X_scvi"] = latent
        sc.pp.neighbors(test_i, use_rep="X_scvi")
        # resolution 1.2 for bcell data, 0.5 for rest
        sc.tl.leiden(test_i, key_added="leiden_scvi", resolution=res)
        pred = test_i.obs["leiden_scvi"].to_list()
        pred = [int(x) for x in pred]
        scores_leiden = clustering_scores(test_i.obs["labels"], pred)
        print(scores_leiden)
        sc.tl.umap(test_i)
        sc.pl.umap(test_i, color=["leiden_scvi", "labels"], title=["result hybridVAE PBMC", "true labels PBMC"])
        results.extend([scores_leiden[0], scores_leiden[1]])
        average_nmi = average_nmi + scores_leiden[0] 
        average_ari = average_ari + scores_leiden[1]

    # Calculate the average over K-folds 
    average = [average_nmi/(K), average_ari/(K)]
    # add results to files
    f.write(str(parameters) + " " + str(average) + " \n")
    f2.write(str(parameters) + " " + str(results) + " \n")
    f.close()
    f2.close()
    print("parameters tested: ", parameters, "average: ", average, "result each fold: ", results)
    return results

def cross_valid_scvi(learning_rate, hidden_layers, size_hidden_layer, data, K, filename, res):
    results = []
    average_nmi = 0
    average_ari = 0
    parameters = [learning_rate, hidden_layers, size_hidden_layer]
    f = open("cross_valid_results/" + filename + "average_results.txt","a")
    f2 = open("cross_valid_results/" + filename + "results.txt","a")
    for i in range(K):
        settings.seed=0
        train_i = copy.deepcopy(data)
        test_i = train_i[i]
        train_i.pop(i)
        train_i = concatenate_adatas(train_i)
        _setup_anndata(test_i, labels_key="labels")
        _setup_anndata(train_i, labels_key="labels")
        model_ = model.SCVI(adata=train_i, n_hidden=size_hidden_layer, n_layers=hidden_layers)
        model_.train(lr=learning_rate)
        latent = model_.get_latent_representation(adata=test_i, hybrid=False)
        test_i.obsm["X_scvi"] = latent
        sc.pp.neighbors(test_i, use_rep="X_scvi")
        sc.tl.leiden(test_i, key_added="leiden_scvi", resolution=res)
        pred = test_i.obs["leiden_scvi"].to_list()
        pred = [int(x) for x in pred]
        scores_leiden = clustering_scores(test_i.obs["labels"], pred)
        results.extend([scores_leiden[0], scores_leiden[1]])
        average_nmi = average_nmi + scores_leiden[0] 
        average_ari = average_ari + scores_leiden[1]

    # Calculate the average over K-folds 
    average = [average_nmi/(K), average_ari/(K)]
    # add results to files
    f.write(str(parameters) + " " + str(average) + " \n")
    f2.write(str(parameters) + " " + str(results) + " \n")
    f.close()
    f2.close()
    print("parameters tested: ", parameters, "average: ", average, "result each fold: ", results)
    return results
    

def start_cross_valid(model_type ,line_nr, gene_indexes_von_mises, data_cross, K_cross, filename, res):
    file = "input/parameters.in"
    f = open(file, "r")
    lines = f.readlines()
    line = lines[line_nr].split()
    learning_rate = float(line[0])
    hidden_layers = int(line[1])
    size_hidden_layer = int(line[2])
    if model_type=="hybrid":
        _ = cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data_cross, K_cross, filename, res)
    if model_type=="scvi": 
        _ = cross_valid_scvi(learning_rate, hidden_layers, size_hidden_layer, data_cross, K_cross, filename, res)
    f.close()

def data_bcell():
    rna = sc.read_h5ad('data/rawRNA.h5ad')
    prna = sc.read_h5ad('data/processedRNA.h5ad')
    # cells with common id
    mdata = mu.MuData({"raw_rna": rna, "processed_rna": prna})
    # remove cells that do not match between prna and rna:
    mu.pp.intersect_obs(mdata)
    K_cross = 3
    adata = mdata['raw_rna']
    sc.pp.filter_cells(adata, min_genes=20)  
    sc.pp.filter_genes(adata, min_cells=3)
    adata.obs.merged.cat.rename_categories({'CD11C+ MBC': 'ITGAX+ MBC', }, inplace= True)
    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    adata.var_names = adata.var_names.str.upper()
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    adata.var["von_mises"] = "false"
    adata.var.loc[cell_cycle_genes, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]
    le = preprocessing.LabelEncoder()
    le.fit(mdata["processed_rna"].obs["merged"])
    labels = le.transform(mdata["processed_rna"].obs["merged"])
    adata.obs["labels"] = labels
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata
    divided_data = divide_data_without_setup(adata, 3)
    data_cross = divide_data_without_setup(divided_data[0],3)
    adata_model = concatenate_adatas([divided_data[1], divided_data[2]])
    return gene_indexes_von_mises, data_cross, K_cross, adata_model

       
def data_cortex():
    adata = _load_cortex(run_setup_anndata=False)
    sc.pp.filter_cells(adata, min_genes=20)
    sc.pp.filter_genes(adata, min_cells=3)
    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    adata.var_names = adata.var_names.str.upper()
    genes = [x for x in adata.var_names]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
    adata.var["von_mises"] = "false"
    for gene in cell_cycle_genes:
        adata.var.loc[adata.var_names == gene, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]
    data_ = divide_data_without_setup(adata, 3)
    # double check how much data in each 
    data_cross = data_[0], data_[1]
    adata_model = data_[2]
    return gene_indexes_von_mises, data_cross, adata_model#adata_train_best_model, adata_test_best_model

def data_pbmc():
    adata = _load_pbmc_dataset(run_setup_anndata=False)
    K_cross = 5
    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    adata.var["gene_symbols"] = adata.var["gene_symbols"].str.upper()
    genes = [x for x in adata.var["gene_symbols"]]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
    adata.var["von_mises"] = "false"
    for gene in cell_cycle_genes:
        adata.var.loc[adata.var["gene_symbols"] == gene, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]
    data_ = divide_data_without_setup(adata, 2)
    data_cross = divide_data_without_setup(data_[0], K_cross)
    adata_model = data_[1]
    return gene_indexes_von_mises, data_cross, K_cross, adata_model

# code for running the cross-validation, switch data_bcell for another dataset
gene_indexes_von_mises, data_cross, K_cross, _ = data_pbmc()
for i in range(40):
    start_cross_valid("hybrid", i, gene_indexes_von_mises,data_cross, K_cross, "cross_valid_hybrid_pbmc_", 0.5)
# running the final cross_valid model with optimal hyperparameters 
K = 5

# gene_indexes_von_mises_bcell, _, _, model_data = data_bcell(
# data_bcell_ = divide_data_without_setup(model_data, K)
# results_hybrid_bcell = cross_valid_hybrid(0.0004, 2, 64, gene_indexes_von_mises_bcell, data_bcell_, K, "bcell_final_test_hybridVI_", 1.2)
# results_scVI_bcell = cross_valid_scvi(0.0003, 1, 128, data_bcell_, K, "bcell_final_test_scvi_", 1.2)
# print("wilcoxon_score_bcell: ", wilcoxon(x=results_hybrid_bcell, y=results_scVI_bcell))

# gene_indexes_von_mises_cortex, _, model_data = data_cortex()
# data_cortex_ = divide_data_without_setup(model_data, K)
# results_hybrid_cortex = cross_valid_hybrid(0.0006, 1, 256, gene_indexes_von_mises_cortex, data_cortex_, K, "cortex_test_hybridVI_", 0.5)
# results_scVI_cortex = cross_valid_scvi(0.0004, 1, 128, data_cortex_, K, "cortex_test_scvi_", 0.5)
# print("wilcoxon_score_cortex: ", wilcoxon(x=results_hybrid_cortex, y=results_scVI_cortex))

# gene_indexes_von_mises_pbmc, _, _, model_data = data_pbmc()
# data_pbmc_ = divide_data_without_setup(model_data, K)
# results_hybrid_pbmc = cross_valid_hybrid(0.0001, 2, 256, gene_indexes_von_mises_pbmc, data_pbmc_, K, "pbmc_final_test_hybridVI_", 0.5)
#results_scvi_pbmc = cross_valid_scvi(0.0004, 1, 128, data_pbmc_, K, "pbmc_final_test_scVI", 0.5)
#print("wilcoxon_score_pbmc: ", wilcoxon(x=results_hybrid_pbmc, y=results_scvi_pbmc))

# # get the combined Wilcoxon results: 
# results = results_hybrid_cortex.extend(results_hybrid_pbmc)
# combined_results_hybrid = results_hybrid_bcell.extend(results)
# results_ = results_scVI_cortex.extend(results_scvi_pbmc)
# combined_results_scVI = results_scVI_bcell.extend(results)
# print("combined wilcoxon score: ", wilcoxon(x=combined_results_hybrid, y=combined_results_scVI))

