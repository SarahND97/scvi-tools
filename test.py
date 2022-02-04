import scanpy as sc
import sys
sys.path.append("/Users/sarahnarrowedanielsson/Documents/KTH/Exjobb/svci-tools") 
# import hybridvi
import scvi.model as model_
import scvi
# import scvi.module as module
from os import path

import math
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

def concatenate_adatas(list_adata):
    return anndata.AnnData.concatenate(*list_adata,batch_key='batch')

datasets =[
    "SRR11816791",
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
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

scvi.data.setup_anndata(
    adata,
    layer="counts"
)

today = date.today()
now = datetime.now()
# dd/mm/YY
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
name = current_date+current_time
name = "04_02_202212_49"

model = scvi.model.HYBRIDVI(adata)
if (path.exists("saved_model/"+name+"hybridvae.model.pkl")):
    model = torch.load('saved_model/'+name+'hybridvae.model.pkl')
else:
    model = scvi.model.HYBRIDVI(adata)
    model.train()
    torch.save(model,'saved_model/'+name+'hybridvae.model.pkl')
latent = model.get_latent_representation()
adata.obsm["scvi"] = latent

adata.obsm["X_pca"] = latent

plt.figure(figsize=(10,8))
plt.suptitle("latent space in hybrid-VAE")
plt.subplot(121)
plt.hist(latent[:,0], bins =100)
plt.subplot(122)
plt.hist(latent[:,1], bins =100)
plt.show()