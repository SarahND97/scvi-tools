import scanpy as sc
from pathlib import Path
import math
import scvi
from os import path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import anndata
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

import pandas as pd
import torch
from datetime import datetime
from datetime import date

def read_one_trust4(adata: anndata.AnnData, basedir: Path, batch: str):
    file_t4 = basedir / "trust4" / batch / "TRUST_possorted_genome_bam_barcode_report.tsv"
    if not file_t4.exists():
        file_t4 = basedir / "trust4" / batch / "TRUST_gex_possorted_bam_barcode_report.tsv"

    if file_t4.exists():
        print("Found trust4 data: "+str(file_t4))
        # Carry over the cell types. Fill with "" if unknown
        df = pd.read_csv(file_t4,sep="\t")
        d = dict(zip(df["#barcode"], df["cell_type"]))
        adata.obs["trust4_celltype"] = [d[cellid] if cellid in d else "" for cellid in adata.obs["cellbc"]]
        return adata
    else:
        return adata

def read_cr(basedir: Path, samplemeta: pd.DataFrame, list_samples=None):

    # Read donor info
    donor_file = basedir / "all_matched_donor_ids.tsv"
    if donor_file.exists():
        print("Found donor information")
        di = pd.read_csv(donor_file)
    else:
        di = None

    # Type of data?
    if (basedir / "cellranger").is_dir():
        cr_dir = (basedir / "cellranger")
    elif (basedir / "cellranger-arc").is_dir():
        cr_dir = (basedir / "cellranger-arc")
    else:
        raise "Unknown cellranger dir"

    # Loop over 10x wells
    if list_samples is None:
        list_samples = [f for f in cr_dir.iterdir() if f.is_dir()]
    list_adata=[]
    for cursamp in list_samples:
        batch=cursamp.name
        print(batch)

        # Load the count matrix for this sample
        adata = sc.read_10x_h5(cursamp / "outs/filtered_feature_bc_matrix.h5")
        adata.var_names_make_unique()
        adata.obs["batchname"] = batch

        # These fields are referenced elsewhere. Putting them here keeps them safe
        adata.obs["batchname"] = batch
        adata.obs["cellbc"] = adata.obs.index

        # Extract donor information for this sample
        if not di is None:
            di_sub=di[di["batch"]==batch]
            map_cell2donor = dict(zip(di_sub["cell"],di_sub["donor_id"]))
            map_cell2prob_doublet = dict(zip(di_sub["cell"],di_sub["prob_doublet"]))
            adata.obs["donor"] = [map_cell2donor[c] for c in adata.obs.index]
            adata.obs["prob_doublet"] = [map_cell2prob_doublet[c] for c in adata.obs.index]

        # Add per-well metadata
        if not samplemeta is None:
            one_samplemeta = samplemeta[samplemeta["batch"]==batch].to_dict(orient="list")
            for key,value in one_samplemeta.items():
                adata.obs[key]=value[0]

        # Add trust4 data
        adata = read_one_trust4(adata, basedir, batch)

        # Add to the collection
        list_adata.append(adata)

    return list_adata



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

adata_v2 = concatenate_adatas(list_adata)

datasets =[
    "SRR11827034",
    "SRR11827035",
    "SRR11827036",
    "SRR11827037",
    #"SRR11827038",  #says missing. hmmmmm would be nice to have!
    "SRR11827039",
]
list_adata=[]
for i in range(len(datasets)):
    print(i)
    fname = "/corgi/cellbuster/holmes2020/cellranger/"+datasets[i]+"/outs/filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(fname)
    adata.var_names_make_unique()
    adata.obs["dataset"] = datasets[i]
    adata.obs["tech"] = "v2rna"
    list_adata.append(adata)
    
adata_holmes = concatenate_adatas(list_adata)

############################ Stewart data ###########################
datasets =[
    "classical",
    "dn",
    "HB34",
    "HB78",
    "igmmem",
    "naive",
    "trans",
]

list_adata=[]
for i in range(len(datasets)):
    print(i)
    fname = "/corgi/cellbuster/stewart2021/processed/"+datasets[i]+"_filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(fname)
    adata.var_names_make_unique()
    adata.obs["dataset"] = datasets[i]
    adata.obs["tech"] = "3prime"
    list_adata.append(adata)
    
adata_stewart = concatenate_adatas(list_adata)

list_adata=read_cr(Path("/corgi/cellbuster/bigb"), samplemeta=None)

adata_cellbuster = concatenate_adatas(list_adata)
print("about to concatenate adatas")
list_adata = [adata_cellbuster, adata_stewart, adata_holmes, adata_v2]
adata = concatenate_adatas(list_adata)
print("concatenated adatas")

# Might need to add telomers 

sc.pp.filter_cells(adata, min_genes=20)  #lower than usual
sc.pp.filter_genes(adata, min_cells=3)

adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4) # here we normalize data 
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

import sys
sys.path.append("/corgi/SarahND/svci-tools") 
scvi.data.setup_anndata(
    adata,
    layer="counts"
)

today = date.today()
now = datetime.now()
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
name = "_" + current_date + "_" + current_time + "_" 
# name = "_08_02_2022_14_14_"

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

def clustering(latent):
    sc.pp.neighbors(latent)
    sc.tl.leiden(latent, key_added = "leiden_1.0")

plt.figure(figsize=(10,8))
plt.suptitle("latent space in hybrid-VAE, n_latent=2")
plt.subplot(121)
plt.hist(latent[:,0], bins =100)
plt.subplot(122)
plt.hist(latent[:,1], bins =100)
fname = "data/latent" + name + ".png"
plt.savefig(fname)
# plt.show()
# clustering(latent)

# import leidenalg
# import igraph as ig
# G = ig.Graph.Erdos_Renyi(100, 0.1);
# For simply finding a partition use:
# part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition);




