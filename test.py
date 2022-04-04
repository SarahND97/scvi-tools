import scanpy as sc
# import sys
# sys.path.append("/Users/sarahnarrowedanielsson/Documents/KTH/Exjobb/svci-tools") 
# import hybridvi
from scvi.data._anndata import _setup_anndata # import scvi.module as module
from scvi import model
from os import path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import anndata

# matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

import pandas as pd
import torch
from datetime import datetime
from datetime import date

def concatenate_adatas(list_adata):
    return anndata.AnnData.concatenate(*list_adata,batch_key='batch')

datasets =[
    "SRR11816791",
    "SRR11816792"
]

list_adata=[]
for i in range(len(datasets)):
    # fname = "data/"+datasets[i]+"/filtered_feature_bc_matrix.h5"
    fname = "/corgi/cellbuster/holmes2020/cellranger/"+datasets[i]+"/outs/filtered_feature_bc_matrix.h5"
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

# Find 
cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
print(cell_cycle_genes)
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
print("adata.var", adata.var_names)
# print("adata.obs", adata.obs)
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
print("cell_cycle_genes:", len(cell_cycle_genes))
# print(adata.var['gene_symbols'][0])
# print(adata.var_names)
adata.var["von_mises"] = "false"
# bad practice need to change
adata.var.loc[cell_cycle_genes, "von_mises"] = "true"
gene_indexes_von_mises = (np.where(adata.var['von_mises'] == "true")[0])
print(len(gene_indexes_von_mises))
_setup_anndata(
    adata,
    layer="counts"
)
today = date.today()
now = datetime.now()
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
name = "_" + current_date + "_" + current_time + "_"
# name = "_07_03_2022_10_30_" # + "only_sperical" + "_"

model_ = model.HYBRIDVI(adata, gene_indexes_von_mises)
if (path.exists("saved_model/"+name+"hybridvae.model.pkl")):
    model_ = torch.load('saved_model/'+name+'hybridvae.model.pkl')
else:
    model_ = model.HYBRIDVI(adata, gene_indexes_von_mises)
    model_.train(lr=0.001)     
    torch.save(model,'saved_model/'+name+'hybridvae.model.pkl')

latent = model_.get_latent_representation()
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=10)
sc.tl.leiden(adata, key_added="leiden_hybridVI", resolution=0.5)
sc.pl.umap(adata, color=["leiden_hybridVI"], save="output/latent_space.png")
