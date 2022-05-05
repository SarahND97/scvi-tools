import scanpy as sc
from pathlib import Path
from scvi import data
from scvi import model
from os import path
import anndata
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from datetime import date
from sklearn.metrics import silhouette_score

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

list_adata=read_cr(Path("/corgi/cellbuster/bigb"), samplemeta=None)

print("concatenate adatas")
adata = concatenate_adatas(list_adata)
print("done")

sc.pp.filter_cells(adata, min_genes=20)  #lower than usual
sc.pp.filter_genes(adata, min_cells=3)

adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4) # here we normalize data 
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

import sys
sys.path.append("/corgi/SarahND/svci-tools") 

sc.pp.filter_cells(adata, min_genes=20)  #lower than usual
sc.pp.filter_genes(adata, min_cells=3)
adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4) # here we normalize data 
# sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`
# Find cell cycle genes
gene_indexes_von_mises = []
gene_indexes_von_mises.extend(np.where(adata.var['von_mises-1'] == "true")[0])
gene_indexes_von_mises.extend(np.where(adata.var['von_mises-2'] == "true")[0])
gene_indexes_von_mises.extend(np.where(adata.var['von_mises-3'] == "true")[0])
gene_indexes_von_mises = np.unique(gene_indexes_von_mises)
print(len(gene_indexes_von_mises))
data.setup_anndata(
    adata,
    layer="counts",
    batch_key = "batch"
)
today = date.today()
now = datetime.now()
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
name = "_" + current_date + "_" + current_time + "_"
# temp: print to see what adata looks like: 

# Hyperparameters from cross-validation result on pbmc
# model_ = model.HYBRIDVI(adata=adata, gene_indexes=gene_indexes_von_mises, n_hidden=256, n_layers=2)
# if (path.exists("saved_model/"+name+"hybridvae.model.pkl")):
#     model_ = torch.load('saved_model/'+name+'hybridvae.model.pkl')
# else:
model_ = model.HYBRIDVI(adata=adata, gene_indexes=gene_indexes_von_mises, n_hidden=256, n_layers=2)
model_.train(lr=0.0001)     
torch.save(model_,'saved_model/'+name+'hybridvae.model.pkl')
# model_ = torch.load('saved_model/'+'_03_05_2022_09_59_hybridvae'+'.model.pkl')

latent = model_.get_latent_representation(hybrid=True)
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata, key_added="leiden_hybridVI", resolution=0.8)
pred = adata.obs["leiden_hybridVI"].to_list()
pred = [int(x) for x in pred]
print(adata.obs)
print("silhouette score: ", silhouette_score(latent, pred))
