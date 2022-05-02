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
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
print("cell_cycle_genes:", len(cell_cycle_genes))
adata.var["von_mises"] = "false"
# bad practice need to change
adata.var.loc[cell_cycle_genes, "von_mises"] = "true"
gene_indexes_von_mises = (np.where(adata.var['von_mises'] == "true")[0])
data.setup_anndata(
    adata,
    layer="counts"
)
today = date.today()
now = datetime.now()
current_date = today.strftime("%d_%m_%Y")
current_time = now.strftime("%H_%M")
name = "_" + current_date + "_" + current_time + "_"
# temp: print to see what adata looks like: 
print(adata.obs)
print(adata.var)

# Hyperparameters from cross-validation result on pbmc
# model_ = model.HYBRIDVI(adata=adata, gene_indexes=gene_indexes_von_mises, n_hidden=256, n_layers=2)
# if (path.exists("saved_model/"+name+"hybridvae.model.pkl")):
#     model_ = torch.load('saved_model/'+name+'hybridvae.model.pkl')
# else:
model_ = model.HYBRIDVI(adata=adata, gene_indexes=gene_indexes_von_mises, n_hidden=256, n_layers=2)
model_.train(lr=0.0001)     
torch.save(model_,'saved_model/'+name+'hybridvae.model.pkl')

latent = model_.get_latent_representation(hybrid=True)
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata, key_added="leiden_hybridVI", resolution=0.5)
pred = adata.obs["leiden_hybridVI"].to_list()
pred = [int(x) for x in pred]
print("silhouette score: ", silhouette_score(pred, adata.obs["labels"]))
diff_exp = model_.differential_expression()
print(diff_exp.head())


# Taken from: https://colab.research.google.com/drive/1V4BD3SAGDwLzvMUn90FMOHYVNG_iP4Ee?usp=sharing#scrollTo=tSuJcKJAfuZx
# markers = {}
# cats = adata.obs.cell_types.cat.categories
# for i, c in enumerate(cats):
#     cid = "{} vs Rest".format(c)
#     cell_type_df = full_de_res.loc[full_de_res.comparison == cid]
#     cell_type_df = cell_type_df.sort_values("lfc_mean", ascending=False)

#     # those genes with higher expression in group 1
#     cell_type_df = cell_type_df[cell_type_df.lfc_mean > 0]

#     # significance
#     cell_type_df = cell_type_df[cell_type_df["bayes_factor"] > 3]
#     # genes with sufficient expression
#     cell_type_df = cell_type_df[cell_type_df["non_zeros_proportion1"] > 0.1]

#     markers[c] = cell_type_df.index.tolist()[:3]

