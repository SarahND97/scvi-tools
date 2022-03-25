from scvi import model
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from scvi.data._built_in_data._pbmc import _load_pbmc_dataset
from scvi.data._anndata import _setup_anndata
import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans
import random

def clustering_scores(labels_true, labels_pred):
    return [NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]

def calculate_average(results):
    averages = {}
    for key, values in results.items():
        average_nmi = 0
        average_ari = 0
        for val in values:
            average_nmi = average_nmi + val[0] + val[2]
            average_ari = average_ari + val[1] + val[3]
        average_nmi = average_nmi/(len(values)*2)
        average_ari = average_ari/(len(values)*2)        
        averages[key] = [average_nmi, average_ari]
    return averages
        


# Find parameters 
# the parameters are learning rate, layers, size of layers, amount of seperation
# need to split data according to labels
def cross_valid_hybrid(parameters, data, K, separation_size):
    results = {}
    for i in range(K):
        data_ = list(data)
        test_i = data_[i]
        train_i = data_.pop(i)
        f = open("output/average_results.txt","w")
        f2 = open("output/results.txt","w")
        for lr in parameters[0]:
            for hidden in parameters[1]:
                for hidden_size in parameters[2]:
                    for ii in range(len(parameters[3])):
                            print("Parameters: ",lr, hidden, hidden_size, ii)
                            model_ = model.HYBRIDVI(adata=train_i, gene_indexes=parameters[3][ii], n_hidden=hidden_size, n_layers=hidden)
                            model_.train(lr=lr)
                            latent = model_.get_latent_representation(adata=test_i, hybrid=True)
                            test_i.obsm["X_scvi"] = latent
                            sc.pp.neighbors(test_i, n_neighbors=20, n_pcs=40, use_rep="X_scvi")
                            sc.tl.leiden(test_i, key_added="leiden_scvi", resolution=0.8)
                            pred = test_i.obs["leiden_scvi"].to_list()
                            pred = [int(x) for x in pred]
                            scores_leiden = clustering_scores(test_i.obs["labels"], pred)
                            labels_pred = KMeans(9, n_init=200).fit_predict(latent)
                            scores_kmeans = clustering_scores(test_i.obs["labels"], labels_pred)
                            para_setting = str([lr, hidden, hidden_size, separation_size[ii]])
                            result = [scores_leiden[0], scores_leiden[1], scores_kmeans[0], scores_kmeans[1]]
                            if para_setting in results:
                                results[para_setting].append(result)
                            else: 
                                results[para_setting] = [result]
    average = calculate_average(results)
    # write file
    f.write( str(average) )
    f2.write( str(results) )
    # close file
    f.close()
    f2.close()
    print(average)


# Model for finding the clustering_scores on the test data
K_cross = 3
adata = _load_pbmc_dataset(run_setup_anndata=False)

# Find 
# cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
# print(cell_cycle_genes
# training is done per cell and not per gene, have to mark entire cells
# s_genes = cell_cycle_genes[:43]
# g2m_genes = cell_cycle_genes[43:]
# cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var["gene_symbols"]]
# print("cell_cycle_genes", len(cell_cycle_genes))
# print(adata.var['gene_symbols'][0])
# print(adata.var_names)

# random marking of cells into either von Mises or Gaussian latent space 
adata.var["von_mises"] = "false"
# random.seed(10)
seperation_size = [5,6,7]
gene_indexes_von_mises = [] 
f =  open("output/indexes_von_mises.txt","w")
indexes = {}
for s in seperation_size:
    genes_vM = random.sample(list(set(adata.var_names)), int(len(list(set(adata.var_names)))/s))
    # bad practice need to change
    adata.var.loc[genes_vM, "von_mises"] = "true"
    gene_indexes_von_mises.append(np.where(adata.var['von_mises'] == "true")[0])
    indexes[str(s)] = [np.where(adata.var['von_mises'] == "true")[0]]
f.write(str(indexes))
f.close()

print("split data")
# TODO: make sure all labels are present in each split, split data into two at first
adata_cross = adata[:int((len(adata)*2)/3), :]
adata_train_best_model = adata[int((len(adata)*2)/3):, :]
size = int(len(adata_cross)/K_cross)
adata_1 = adata_cross[:size,:]
adata_2 = adata_cross[size:2*size,:]
adata_3 = adata_cross[2*size:,:]
data = [adata_train_best_model, adata_1, adata_2, adata_3]
for d in range(len(data)):
    da = data[d].copy()
    _setup_anndata(da, batch_key="batch", labels_key="labels")
    data[d] = da 
    # Check the distribution of the labels
    # for i in range(9):
    #     print(len(np.where(data[d].obs['labels'] == i)[0]))
data_cross = [data[1], data[2], data[3]]
# learning_rate = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
learning_rate = [0.002,0.003, 0.004]
hidden_layers = [1,2,3]
size_hidden_layer = [64,128,256]
parameters = [learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises]
cross_valid_hybrid(parameters, data_cross, K_cross, seperation_size)

# if (path.exists("saved_model/"+name+"hybrid.model.pkl")):
#     model_ = torch.load('saved_model/'+name+'hybrid.model.pkl')
# else:
#     model_ = model.HYBRIDVI(adata, gene_indexes=gene_indexes_von_mises)
#     model_.train()     
#     torch.save(model_,'saved_model/'+name+'SCVI.model.pkl')

# results ={'[0.002, 1, 64, 5]': [[0.6332638964787377, 0.3012328464408762, 0.716215250878572, 0.622744812497254], 
# [0.6204473762953144, 0.2927482711560561, 0.6927789928318266, 0.5594955515992779], 
# [0.6671471670705262, 0.31661730059121906, 0.7837582290367187, 0.6512656772937997]], '[0.003, 3, 64, 5]': [[0.6332638964787377, 0.3012328464408762, 0.716215250878572, 0.622744812497254], 
# [0.6204473762953144, 0.2927482711560561, 0.6927789928318266, 0.5594955515992779], 
# [0.6671471670705262, 0.31661730059121906, 0.7837582290367187, 0.6512656772937997]]}
# print(calculate_average(results))