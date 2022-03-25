import scanpy as sc
import sys
sys.path.append("/Users/sarahnarrowedanielsson/Documents/KTH/Exjobb/svci-tools") 
# import hybridvi
import scvi
# import scvi.module as module
from os import path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scanpy

import anndata

# matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

import pandas as pd
import torch
from datetime import datetime
from datetime import date

# cmap_cellcycle = matplotlib.colors.LinearSegmentedColormap.from_list(
#     "cellcycle", ["yellow", "green", "magenta", "yellow"], N=128)

# # cell cycle genes
# cc_mouse = pd.read_csv("data/cc_mouse.csv")
# cc_mouse = cc_mouse.iloc[:,1:]
# cc_mouse.head()
# cell_cycle_genes = [g.upper() for g in cc_mouse["symbol"].tolist()]

# ### Real data
# adata = sc.read("data/small_rna.h5ad")
# # fname = "data/classical_filtered_feature_bc_matrix.h5"
# # adata = sc.read_10x_h5(fname)
# adata.var_names_make_unique()
# sc.pp.filter_genes(adata, min_counts=3)

# adata.layers["counts"] = adata.X.copy() # preserve counts
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# adata.raw = adata 

# sc.pp.highly_variable_genes(
#     adata,
#     n_top_genes=1200,
#     subset=True,
#     layer="counts",
#     flavor="seurat_v3"
# )

# adata.var["highly_variable"] = [True if g in cell_cycle_genes else False for g in adata.var.index.tolist()]
# adata = adata[:,adata.var["highly_variable"]]

# adata = adata.copy()

# scvi.data.setup_anndata(
#     adata,
#     layer="counts",
# )

# model = ""


# # if (path.exists("saved_model/")):
# #     model = scvi.model.HYBRIDVI("saved_model/", adata, use_gpu=False)
# # else:
# model = scvi.model.HYBRIDVI(adata, n_latent=2)
# model.train()
# # model.save("saved_model/")

# adata_PBMC = sc.read_10x_mtx(
#     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
#     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
#     cache=True) 

# adata_PBMC.var_names_make_unique()

def concatenate_adatas(list_adata):
    return anndata.AnnData.concatenate(*list_adata,batch_key='batch')

datasets =[
    #"SRR11816791",
    "SRR11816792"
]

# /corgi/cellbuster/holmes2020/cellranger/"+datasets[i]+"/outs/filtered_feature_bc_matrix.h5

list_adata=[]
for i in range(len(datasets)):
    fname = "data/"+datasets[i]+"/filtered_feature_bc_matrix.h5"
    # fname = "/corgi/cellbuster/holmes2020/cellranger/"+datasets[i]+"/outs/filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(fname)
    adata.var_names_make_unique()
    adata.obs["dataset"] = datasets[i]
    adata.obs["tech"] = "v2rna"
    list_adata.append(adata)
    
adata = concatenate_adatas(list_adata)

sc.pp.filter_cells(adata, min_genes=20)  #lower than usual
sc.pp.filter_genes(adata, min_cells=3)

adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4) # here we normalize data 
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

scvi.data.setup_anndata(
    adata,
    layer="counts"
)

today = date.today()
now = datetime.now()
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
# name = "_" + current_date + "_" + current_time + "_"
name = "_07_03_2022_10_30_" # + "only_sperical" + "_"

model = scvi.model.HYBRIDVI(adata)
if (path.exists("saved_model/"+name+"hybridvae.model.pkl")):
    model = torch.load('saved_model/'+name+'hybridvae.model.pkl')
else:
    model = scvi.model.HYBRIDVI(adata)
    model.train()     
    torch.save(model,'saved_model/'+name+'hybridvae.model.pkl')

latent = model.get_latent_representation()
# print(latent.shape)
# print(latent[0].shape)

# adata.obsm["X_pca"] = latent

# plt.figure(figsize=(10,8))
# plt.suptitle("latent space in hybrid-VAE, n_latent=2")
# plt.subplot(121)
# plt.hist(latent[:,0], bins =100)
# plt.subplot(122)
# plt.hist(latent[:,1], bins =100)
# plt.show()
# sc.tl.pca(adata)
# sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# sc.tl.leiden(adata, resolution=2)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color = ["leiden"])
#sc.tl.pca(latent)
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=20)
#sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, key_added="leiden_hybridVI", resolution=0.8)
pred = adata.obs['leiden_hybridVI'].to_list()
pred = [int(x) for x in pred]
print(np.unique(pred).shape)
# sc.pl.umap(adata, color=["leiden_hybridVI"])
# sc.pp.neighbors(latent, n_neighbors=10, n_pcs=40)
# sc.tl.leiden(latent, resolution=2)
# sc.tl.umap(latent)
# sc.pl.umap(latent, color = ["leiden"])

def show_tSNE(tsne, name):
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1])    

    plt.axis("off")
    plt.title("tSNE for latent space " + name, fontsize=16)
    plt.show()

# tsne = TSNE().fit_transform(latent)
# show_tSNE(tsne, "hybrid-scVI")
#print(tsne[:, 0], tsne[:, 1])
# original paper uses k-means clustering 
# sc.pp.neighbors(adata, n_neighbors=5, n_pcs=10)
# sc.tl.leiden(adata, resolution=2)
# sc.pl.umap(adata,color=["leiden","cellcycle","batchname",
#                         "donor","trust4_celltype"],ncols=2,legend_loc="on data")
# import leidenalg
# import igraph as ig
# edgelist = zip(list(latent[0]), list(latent[1]))
# graph = ig.Graph(edgelist)
# partition = leidenalg.find_partition(latent, leidenalg.ModularityVertexPartition)
# print(partition)

# def clustering(K):
#     # reduced_data = PCA(n_components=2).fit_transform(data)
#     reduced_data = adata.obsm["X_pca"] 
#     # reduced_data = PCA(2).fit_transform(reduced_data)
#     # print(reduced_data)
#     print(reduced_data.shape)
#     # print(latent_space.shape)
#     kmeans = KMeans(n_clusters=K, n_init=200) # .fit(latent_space)
#     labels = kmeans.fit_predict(reduced_data)
#     # np.where(labels_array == clustNum)[0]
#     # print(np.where(labels==0)[0])

#     print(labels)
#     u_labels = np.unique(labels)
#     # Plotting the results:
#     for i in u_labels:
#         plt.scatter(reduced_data[labels == i , 0] , reduced_data[labels == i , 1] , label = i)
#     plt.legend()
#     plt.show()

# vill se vilken gen som är i vilket kluster
# för imorgon 
    

# clustering(10)

# print(latent.shape)
# plt.scatter(latent[:,0] , latent[:,1] , label = i)
# plt.legend()
# plt.show()