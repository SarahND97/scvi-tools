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
import matplotlib

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
    # "SRR11816792"
]

list_adata=[]
cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
for i in range(len(datasets)):
    # fname = "data/"+datasets[i]+"/filtered_feature_bc_matrix.h5"
    fname = "/corgi/cellbuster/holmes2020/cellranger/"+datasets[i]+"/outs/filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(fname)
    adata.var["von_mises"] = "false"
    cell_cycle_genes_von_mises = [x for x in cell_cycle_genes if x in adata.var_names]
    adata.var.loc[cell_cycle_genes_von_mises, "von_mises"] = "true"
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
gene_indexes_von_mises = np.where(adata.var['von_mises'] == "true")[0]

_setup_anndata(
    adata,
    layer="counts"
)
today = date.today()
now = datetime.now()
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
name = "_" + current_date + "_" + current_time + "_"
# name = "_04_04_2022_16_38_" 

model_ = model.HYBRIDVI(adata, gene_indexes_von_mises)
if (path.exists("saved_model/"+name+"hybridvae.model.pt")):
    model_.load('saved_model/'+name+'hybridvae.pt')
    model_.eval()
else:
    model_ = model.HYBRIDVI(adata, gene_indexes_von_mises)
    model_.train(lr=0.001)     
    model_.save(dir_path='saved_model/'+name+'hybridvae.pt', overwrite=True)

# latent = model_.get_latent_representation()
# adata.obsm["X_scVI"] = latent
# sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_scVI")
# sc.tl.leiden(adata, key_added="X_scVI", resolution=0.5)
# sc.pl.umap(adata, color=["X_scVI"], save="output/latent_space.png")
# adata.obsm["X_pca"] = latent[:,10:]
# sc.pp.neighbors(adata)

# cmap_cellcycle = matplotlib.colors.LinearSegmentedColormap.from_list(
#     "cellcycle", ["yellow", "green", "magenta", "yellow"], N=128) 

# import re
# aicda = [i for i,g in enumerate(adata.var.index.tolist()) if g == "AICDA"]
# cxcr4 = [i for i,g in enumerate(adata.var.index.tolist()) if re.search("cxcr4", g, re.IGNORECASE)]
# cd83 = [i for i,g in enumerate(adata.var.index.tolist()) if g == "CD83"]
# ccr6 = [i for i,g in enumerate(adata.var.index.tolist()) if g == "CCR6"]
# cd9 = [i for i,g in enumerate(adata.var.index.tolist()) if g == "CD9"]

# for i,g in enumerate(adata.var.index.tolist()):
#     if re.search("ccr", g, re.IGNORECASE):
#         print(g)

# adata.obsm["AICDA"] = adata.X.todense()[:,int(aicda[0])]
# adata.obsm["CD83"] = adata.X.todense()[:,int(cd83[0])]
# adata.obsm["CCR6"] = adata.X.todense()[:,int(ccr6[0])]
# adata.obsm["CD9"] = adata.X.todense()[:,int(cd9[0])]

# sc.pl.embedding(adata, basis = "X_scSVAE",color = [ "gc_zone", "AICDA", "gc_zone", "CD83", "gc_zone", "CCR6", "gc_zone", "CD9"], ncols = 2,
#                 size=120, color_map=cmap_cellcycle, components = ['11, 10, 12'],
#                projection = '3d')