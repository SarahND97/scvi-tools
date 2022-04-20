from scvi import model
import anndata
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from scvi.data._built_in_data._pbmc import _load_pbmc_dataset
from scvi.data._built_in_data._cortex import _load_cortex
from scvi.data._anndata import _setup_anndata
import scanpy as sc
import numpy as np

def concatenate_adatas(list_adata):
    return anndata.AnnData.concatenate(*list_adata,batch_key='batch')

def create_parameters_file():
    learning_rate = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
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

# Find parameters 
# the parameters are learning rate, layers, size of layers
def cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data, K):
    results = []
    average_nmi = 0
    average_ari = 0
    parameters = [learning_rate, hidden_layers, size_hidden_layer]
    f = open("output/average_results.txt","a")
    f2 = open("output/results.txt","a")
    for i in range(K):
        data_ = list(data)
        test_i = data_[i]
        train_i = data_.pop(i)
        model_ = model.HYBRIDVI(adata=train_i, gene_indexes=gene_indexes_von_mises, n_hidden=size_hidden_layer, n_layers=hidden_layers)
        model_.train(lr=learning_rate)
        latent = model_.get_latent_representation(adata=test_i, hybrid=True)
        test_i.obsm["X_scvi"] = latent
        sc.pp.neighbors(test_i, n_neighbors=10, use_rep="X_scvi")
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

def start_cross_valid(line_nr, gene_indexes_von_mises, data_cross, K_cross):
    file = "input/parameters.in"
    f = open(file, "r")
    lines = f.readlines()
    line = lines[line_nr].split()
    learning_rate = float(line[0])
    hidden_layers = int(line[1])
    size_hidden_layer = int(line[2])
    cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data_cross, K_cross)
    f.close()

def final_result_scvi(train, test):
    learning_rate = 0.0004
    hidden_layers = 1
    size_hidden_layer = 128
    model_ = model.SCVI(adata=train, n_hidden=size_hidden_layer, n_layers=hidden_layers)
    model_.train(lr=learning_rate)
    latent = model_.get_latent_representation(adata=test, hybrid=False)
    test.obsm["X_scvi"] = latent
    sc.pp.neighbors(test, n_neighbors=10, use_rep="X_scvi")
    sc.tl.leiden(test, key_added="leiden_scvi", resolution=0.5)
    pred = test.obs["leiden_scvi"].to_list()
    pred = [int(x) for x in pred]
    scores_leiden = clustering_scores(test.obs["labels"], pred)
    print("final_scores_: " ,scores_leiden[0], scores_leiden[1], (scores_leiden[0] + scores_leiden[1])/2)
       


def final_result_hybrid(dataset, gene_indexes, train, test):
    learning_rate = 0
    hidden_layers = 0 
    size_hidden_layer = 0

    if dataset=="pbmc":
        learning_rate = 0.0001
        hidden_layers = 2
        size_hidden_layer = 256

    elif dataset=="cortex":
        # remember to change these after cross_valid
        learning_rate = 0.0006
        hidden_layers = 1
        size_hidden_layer = 256
    
    model_ = model.HYBRIDVI(adata=train, gene_indexes=gene_indexes, n_hidden=size_hidden_layer, n_layers=hidden_layers)
    model_.train(lr=learning_rate)
    latent = model_.get_latent_representation(adata=test, hybrid=True)
    test.obsm["X_scvi"] = latent
    sc.pp.neighbors(test, n_neighbors=10, use_rep="X_scvi")
    sc.tl.leiden(test, key_added="leiden_scvi", resolution=0.5)
    pred = test.obs["leiden_scvi"].to_list()
    pred = [int(x) for x in pred]
    scores_leiden = clustering_scores(test.obs["labels"], pred)
    print("final_scores_"+dataset+": ",scores_leiden[0], scores_leiden[1], (scores_leiden[0] + scores_leiden[1])/2)
       
def data_cortex():
    adata = _load_cortex(run_setup_anndata=False)
    K_cross = 2

    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    adata.var_names = adata.var_names.str.upper()
    genes = [x for x in adata.var_names]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
    adata.var["von_mises"] = "false"
    for gene in cell_cycle_genes:
        adata.var.loc[adata.var_names == gene, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]

    adata_1 = []
    adata_2 = []
    adata_train_best_model = []
    adata_test_best_model = []
    for label in set(adata.obs["labels"]):
         label_list = adata[np.where(adata.obs["labels"] == label)[0]]
         size = int(len(label_list)/2)
         total = len(label_list)
         adata_1.append(label_list[:int(size/2),:])
         adata_2.append(label_list[int(size/2):size,:])
         adata_train_best_model.append(label_list[size:int(9*total/10),:])
         adata_test_best_model.append(label_list[int(9*total/10):,:])

    adata_1 = concatenate_adatas(adata_1)
    adata_2 = concatenate_adatas(adata_2)
    adata_train_best_model = concatenate_adatas(adata_train_best_model)
    adata_test_best_model = concatenate_adatas(adata_test_best_model)
    data = [adata_1, adata_2, adata_train_best_model, adata_test_best_model]
    for d in range(len(data)):
        da = data[d].copy()
        _setup_anndata(da, labels_key="labels")
        data[d] = da
    print("train")
    _setup_anndata(adata_train_best_model, labels_key="labels")
    print("test")
    _setup_anndata(adata_test_best_model, labels_key="labels")

    return gene_indexes_von_mises, data, K_cross, adata_train_best_model, adata_test_best_model

def data_pbmc():
    adata = _load_pbmc_dataset(run_setup_anndata=False)
    K_cross = 3
    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    genes = [x for x in adata.var["gene_symbols"]]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
    adata.var["von_mises"] = "false"
    for gene in cell_cycle_genes:
        adata.var.loc[adata.var["gene_symbols"] == gene, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]
    adata_cross = adata[:int((len(adata)*2)/K_cross), :]
    size = int(len(adata_cross)/K_cross)
    adata_1 = adata_cross[:size,:]
    adata_2 = adata_cross[size:2*size,:]
    adata_3 = adata_cross[2*size:,:]
    adata_model = adata[int((len(adata)*2)/K_cross):,:]
    adata_train_best_model = []
    adata_test_best_model = []
    for label in set(adata_model.obs["labels"]):
        label_list = adata_model[np.where(adata_model.obs["labels"] == label)[0]]
        total = len(label_list)
        print(total)
        adata_train_best_model.append(label_list[:int(7*total/10),:])
        adata_test_best_model.append(label_list[int(7*total/10):,:])
    
    adata_train_best_model = concatenate_adatas(adata_train_best_model)
    adata_test_best_model = concatenate_adatas(adata_test_best_model)
    
    
    data = [adata_1, adata_2, adata_3, adata_train_best_model, adata_test_best_model]
    for d in range(len(data)):
        da = data[d].copy()
        _setup_anndata(da, batch_key="batch", labels_key="labels")
        data[d] = da 
    data_cross = data[:3]

    return gene_indexes_von_mises, data_cross, K_cross, data[3], data[4]

# gene_indexes_von_mises_cortex, _, _, train_cortex, test_cortex = data_cortex()
gene_indexes_von_mises_pbmc, _, _, train_pbmc, test_pbmc = data_pbmc()
#final_result("cortex", gene_indexes_von_mises_cortex, train_cortex, test_cortex)
# final_result_hybrid("pbmc", gene_indexes_von_mises_pbmc, train_pbmc, test_pbmc)
print("pbmc")
final_result_scvi(train_pbmc, test_pbmc)
#print("cortex")
#final_result_scvi(train_cortex, test_cortex)
# for i in range(48):
#     start_cross_valid(i, gene_indexes_von_mises, data_cross, K_cross)