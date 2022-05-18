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
    return anndata.AnnData.concatenate(*list_adata)

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

def divide_data(data, K):
    divided_data = [[] for _ in range(K)]
    # make sure that there is an equal amount of labels in each dataset
    for i in range(K):
        for label in set(data.obs["labels"]):
            label_list = data[np.where(data.obs["labels"] == label)[0]]
            total = len(label_list)
            size = int(total/K)
            if i<K-1:
                divided_data[i].append(label_list[i*size:size*(i+1),:])
            # the last fold gets the remainder of the cells
            else: 
                divided_data[i].append(label_list[i*size:,:])
    return [concatenate_adatas(d) for d in divided_data]

# Find parameters 
# the parameters are learning rate, layers, size of layers
def cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data, K, filename, res, seed):
    results = []
    average_nmi = 0
    average_ari = 0
    parameters = [learning_rate, hidden_layers, size_hidden_layer]
    f = open(filename + "average_results.txt","a")
    f2 = open(filename + "results.txt","a")
    for i in range(K):
        settings.seed=seed
        train_i = copy.deepcopy(data)
        test_i = train_i[i]
        train_i.pop(i)
        # set up anndata does not work for lists
        if len(data)<2:
            train_i = train_i[0]
        else:
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
        print("parameters: ", parameters, "fold: ", i, "scores: ", scores_leiden)
        results.append([scores_leiden[0], scores_leiden[1]])
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

def cross_valid_scvi(learning_rate, hidden_layers, size_hidden_layer, data, K, filename, res, seed):
    results = []
    average_nmi = 0
    average_ari = 0
    parameters = [learning_rate, hidden_layers, size_hidden_layer]
    f = open(filename + "average_results.txt","a")
    f2 = open(filename + "results.txt","a")
    for i in range(K):
        settings.seed=seed
        train_i = copy.deepcopy(data)
        test_i = train_i[i]
        train_i.pop(i)
        # set up anndata does not work for lists
        if len(data)<2:
            train_i = train_i[0]
        else:
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
        results.append([scores_leiden[0], scores_leiden[1]])
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
    
def start_cross_valid(model_type ,line_nr, gene_indexes_von_mises, data_cross, K_cross, filename, res, seed):
    file = "input/parameters.in"
    f = open(file, "r")
    lines = f.readlines()
    line = lines[line_nr].split()
    learning_rate = float(line[0])
    hidden_layers = int(line[1])
    size_hidden_layer = int(line[2])
    if model_type=="hybrid":
        _ = cross_valid_hybrid(learning_rate, hidden_layers, size_hidden_layer, gene_indexes_von_mises, data_cross, K_cross, filename, res, seed)
    if model_type=="scvi": 
        _ = cross_valid_scvi(learning_rate, hidden_layers, size_hidden_layer, data_cross, K_cross, filename, res, seed)
    f.close()

def visualize_von_mises_fisher(data, latent, title):
    latent[:,8:].shape
    data.obsm["latent"] = latent
    sc.pl.embedding(data, basis = "latent",color = ["labels"], size=120, 
                    components = ['9,10'], title=title)

def data_bcell(K):
    rna = sc.read_h5ad('../../filippe/rawRNA.h5ad')
    prna = sc.read_h5ad('../../filippe/processedRNA.h5ad')
    prna.obs.merged.cat.rename_categories({'CD11C+ MBC': 'ITGAX+ MBC', }, inplace= True)
    # cells with common id
    mdata = mu.MuData({"raw_rna": rna, "processed_rna": prna})
    # remove cells that do not match between prna and rna:
    mu.pp.intersect_obs(mdata)
    adata = mdata['raw_rna']
    sc.pp.filter_cells(adata, min_genes=20)  
    sc.pp.filter_genes(adata, min_cells=3)
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
    divided_data = divide_data(adata, 2)
    data_cross = divide_data(divided_data[0],K)
    adata_model = divided_data[1]
    return gene_indexes_von_mises, data_cross, adata_model

       
def data_cortex(K):
    adata = _load_cortex(run_setup_anndata=False)
    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    adata.var_names = adata.var_names.str.upper()
    genes = [x for x in adata.var_names]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
    adata.var["von_mises"] = "false"
    for gene in cell_cycle_genes:
        adata.var.loc[adata.var_names == gene, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]
    data_ = divide_data(adata, 3)
    # double check how much data in each 
    data_cross = divide_data(data_[0], K)
    adata_model = concatenate_adatas([data_[1], data_[2]])
    return gene_indexes_von_mises, data_cross, adata_model#adata_train_best_model, adata_test_best_model

def data_pbmc(K):
    adata = _load_pbmc_dataset(run_setup_anndata=False)
    # Find the cell cycle genes in the data
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    adata.var["gene_symbols"] = adata.var["gene_symbols"].str.upper()
    genes = [x for x in adata.var["gene_symbols"]]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in genes]
    adata.var["von_mises"] = "false"
    for gene in cell_cycle_genes:
        adata.var.loc[adata.var["gene_symbols"] == gene, "von_mises"] = "true"
    gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]
    data_ = divide_data(adata, 2)
    data_cross = divide_data(data_[0], K)
    adata_model = data_[1]
    return gene_indexes_von_mises, data_cross, adata_model

def run_cross_validation_pbmc(K):
    gene_indexes_pbmc, data_cross_pbmc, _ = data_pbmc(K)
    for i in range(40):
        start_cross_valid("hybrid", i, gene_indexes_pbmc, data_cross_pbmc, K, "cross_valid_results/cross_valid_hybrid_pbmc_", 0.5, 0)

def run_cross_validation_cortex(K):
    gene_indexes_cortex, data_cross_cortex, _ = data_cortex(K)
    for i in range(40):
        start_cross_valid("hybrid", i, gene_indexes_cortex, data_cross_cortex, K, "cross_valid_results/cross_valid_hybrid_cortex_", 0.5, 0)

def run_cross_validation_bcell(K):
    gene_indexes_bcell, data_cross_bcell, _ = data_bcell(K)
    for i in range(40):
        start_cross_valid("hybrid", i, gene_indexes_bcell, data_cross_bcell, K, "cross_valid_results/cross_valid_hybrid_bcell_", 1.2, 0)

def get_wilcoxon_score(K, filename, seeds, results_hybrid, results_scVI):
    average_hybrid = np.zeros((len(seeds),K))
    average_scVI = np.zeros((len(seeds),K))
    nmi_scvi = np.zeros((len(seeds),K))
    ari_scvi = np.zeros((len(seeds),K))
    nmi_hybrid = np.zeros((len(seeds),K))
    ari_hybrid = np.zeros((len(seeds),K))
    print(filename)
    # filename should include path
    f = open(filename + ".txt","a")
    for i in range(len(seeds)):
        for j in range(K):
            nmi_hybrid[i][j] = results_hybrid[i][j][0]
            ari_hybrid[i][j] = results_hybrid[i][j][1]
            nmi_scvi[i][j] = results_scVI[i][j][0]
            ari_scvi[i][j] = results_scVI[i][j][1]
            average_hybrid[i] += results_hybrid[i][j]
            average_scVI[i] += results_scVI[i][j]
    average_hybrid /= len(seeds)
    average_scVI /= len(seeds)
    for i in range(len(seeds)):
        print("Wilcoxon seed " + str(seeds[i]) + ": ", wilcoxon(x=np.ravel(results_hybrid[i]), y=np.ravel(results_scVI[i])))
    wilcoxon_score=wilcoxon(x=np.ravel(average_hybrid), y=np.ravel(average_scVI))
    f.write("Wilcoxon: " + str(wilcoxon_score) + " Hybrid_average: " + str(np.ravel(average_hybrid)) + " scvi_average: " + str(np.ravel(average_scVI)))
    f.close()
    print("wilcoxon_score: ", wilcoxon_score)

    for i in range(K):
        string_nmi_hybrid = "nmi fold " + str(i)
        write_mean_std(nmi_scvi[i],nmi_hybrid[i], string_nmi_hybrid, filename)
        string_ari_scvi = "ari fold " + str(i)
        write_mean_std(ari_scvi[i], ari_hybrid[i],string_ari_scvi, filename)

def write_mean_std(scvi_, hybrid, fold, filename):
    f = open(filename + "folds_standard_deviation_mean" + ".txt","a")
    f.write("standard deviation " + fold + " scvi: " + str(np.std(scvi_)) + " hybrid: " + str(np.std(hybrid))+"\n")
    f.write("mean " + fold + " scvi: " + str(np.mean(scvi_)) + " hybrid: " + str(np.mean(hybrid))+"\n")
    f.close()

def run_wilcoxon_tests():
    # running the model with optimal hyperparameters on the three datasets over 5 random seeds
    seeds = [0,10,20,30,40]
    K=5
    gene_indexes_cortex, _, data_wilcoxon_cortex = data_cortex(K)
    gene_indexes_pbmc, _, data_wilcoxon_pbmc = data_pbmc(K)
    gene_indexes_bcell, _, data_wilcoxon_bcell = data_bcell(K)
    cortex = divide_data(data_wilcoxon_cortex, K)
    pbmc = divide_data(data_wilcoxon_pbmc, K)
    bcell = divide_data(data_wilcoxon_bcell, K)
    results_hybrid_cortex = [] 
    results_scVI_cortex = []
    results_hybrid_pbmc = [] 
    results_scVI_pbmc = []
    results_hybrid_bcell = [] 
    results_scVI_bcell = []
    for i in range(len(seeds)):
        # run cross-valid for cortex data
        results_hybrid_cortex.append(cross_valid_hybrid(0.0004, 2, 256, gene_indexes_cortex, cortex, K, "wilcoxon_results/cross_valid_hybrid_cortex_", 0.5, seeds[i]))
        results_scVI_cortex.append(cross_valid_scvi(0.0004, 1, 128, cortex, K, "wilcoxon_results/cross_valid_scvi_cortex", 0.5, seeds[i]))
        # run cross-valid for pbmc data
        results_hybrid_pbmc.append(cross_valid_hybrid(0.0005, 2, 512, gene_indexes_pbmc, pbmc, K, "wilcoxon_results/cross_valid_hybrid_pbmc_", 0.5, seeds[i]))
        results_scVI_pbmc.append(cross_valid_scvi(0.0004, 1, 128, pbmc, K, "wilcoxon_results/cross_valid_scvi_pbmc", 0.5, seeds[i]))
        # run cross-valid for bcell data
        results_hybrid_bcell.append(cross_valid_hybrid(0.0003, 2, 512, gene_indexes_bcell, bcell, K, "wilcoxon_results/cross_valid_hybrid_bcell_", 0.5, seeds[i]))
        results_scVI_bcell.append(cross_valid_scvi(0.0004, 1, 128, bcell, K, "wilcoxon_results/cross_valid_scvi_bcell_", 0.5, seeds[i]))
    get_wilcoxon_score(K, "wilcoxon_results/cortex_", seeds, results_hybrid_cortex, results_scVI_cortex)
    get_wilcoxon_score(K, "wilcoxon_results/pbmc_", seeds, results_hybrid_pbmc, results_scVI_pbmc)
    get_wilcoxon_score(K, "wilcoxon_results/bcell_", seeds, results_hybrid_bcell, results_scVI_bcell)
    
run_wilcoxon_tests()






