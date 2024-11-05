#import os
# %%
import subprocess
import importlib
from pathlib import Path
import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import glob
import cellxgene_census
import leidenalg
import igraph
import scvi
from scipy.sparse import csr_matrix
import warnings
import cellxgene_census
import cellxgene_census.experimental
import scvi
import tempfile
import botocore
import torch
from scvi.hub import HubModel
import gzip
from sklearn.ensemble import RandomForestClassifier
#datasets = ["lau","lim","nagy","pineda","rosmap","velmeshev"]
#topdir="/space/scratch/ericchu/r_cache/041_CH4_FINAL/data/"
projPath="/space/grp/rschwartz/rschwartz/census-stuff"
import scf
from scf import *
# Set pandas to display all columns
pd.set_option('display.max_columns', None)
CENSUS_VERSION = "2024-07-01"
census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
sc.set_figure_params(figsize=(10, 10), frameon=False)
torch.set_float32_matmul_precision("high")
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
#save_dir = tempfile.TemporaryDirectory()
organism = "homo sapiens"
#query_tissue = "cortex"
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  #

#outdir = organism + - + census_verion
#outdir doesnt exist
   # os.makedirs(outdir) 
    ## Open the CellxGene census
#if "model.pt: file not in outdir:
    #census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

    ## Get scVI model info
    #scvi_info = cellxgene_census.experimental.get_embedding_metadata_by_name(
        #embedding_name="scvi",
        #organism=organism.replace(" ", "_"),  
        #census_version=CENSUS_VERSION,
    #)

    ## Extract the model link
    #model_link = scvi_info["model_link"]
    #date=str.split(model_link,"/")[5]
    #url = os.path.join("https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/",date,organism,"model.pt")

    ## Download the model using wget if it doesn't already exist
    #subprocess.run(["!wget --no-check-certificate -q -O", url, "-O", os.path.join(outdir,"model.pt")])
#else:
    #print(f"File already exists at {outdir}, skipping download.")


# %%
brain_obs = cellxgene_census.get_obs(census, organism,
    value_filter=(
        "tissue_general == 'brain' and "
        "is_primary_data == True and "
        "disease == 'normal'"
    )
)
datasets = census["census_info"]["datasets"].read().concat().to_pandas()
# Concatenates results to pyarrow.Table
brain_obs = brain_obs.merge(datasets, on="dataset_id", suffixes=(None,"y"))
brain_obs_filtered = brain_obs[brain_obs["collection_name"].isin(
        ["SEA-AD: Seattle Alzheimer’s Disease Brain Cell Atlas",
         "Transcriptomic cytoarchitecture reveals principles of human neocortex organization"])]

brain_cell_subsampled_n = 10000
brain_cell_subsampled_ids = brain_obs_filtered["soma_joinid"].sample(brain_cell_subsampled_n, 
                                                                   random_state=1).tolist()
# Set organism name based on the input value
if organism == "homo sapiens":
    organism_name = "Homo sapiens"
elif organism == "mus musculus":
    organism_name = "Mus musculus"
else:
    raise ValueError("Unsupported organism")

adata_census = cellxgene_census.get_anndata(
    census=census,
    measurement_name="RNA",
    organism="Homo sapiens",
   # obs_value_filter=f"dataset_id in {dataset_ids}",
    obs_embeddings=["scvi"],
    obs_coords=brain_cell_subsampled_ids,
)
adata_census.var.set_index("feature_id", inplace=True)
adata_census.obs = adata_census.obs.merge(datasets, on="dataset_id", suffixes=(None,"y"))
#adata_census = adata_census[:, adata_census.var["feature_name"].notnull().values].copy()

# %%
#lau = ad.read_h5ad("/space/grp/rschwartz/rschwartz/census-stuff/h5ad/lau_updated.h5ad") 
lau_sub= ad.read_h5ad("/space/grp/rschwartz/rschwartz/census-stuff/h5ad/lau_updated_subsampled_1000.h5ad")
adata=lau_sub
adata.var["ensembl_id"] = adata.var.index
adata.obs["n_counts"] = adata.X.sum(axis=1)
adata.obs["joinid"] = list(range(adata.n_obs))
adata.obs["batch"] = adata.obs["sample"]
# filter out missing HGNC features
# this ensures features are the same as seurat implementation
adata = adata[:, adata.var["gene_name"].notnull().values].copy()
#adata.var.set_index("gene_name", inplace=True)

#Run them through the scVI forward pass and extract their latent representation (embedding):
scvi.model.SCVI.prepare_query_anndata(adata, os.path.join(projPath, "scvi-human-2024-07-01"))
vae_q = scvi.model.SCVI.load_query_data(
    adata, os.path.join(projPath,
    "scvi-human-2024-07-01")
)
# This allows for a simple forward pass
vae_q.is_trained = True
latent = vae_q.get_latent_representation()
adata.obsm["scvi"] = latent

#adata.obs["cell_type"] = adata.obs["Cell_type"]
adata.obs["tissue"] = "prefrontal cortex"
adata.obs["tissue_general"] = "brain"

sc.pp.neighbors(adata, n_neighbors=30, use_rep="scvi")
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
#sc.pl.umap(adata, color=["leiden","Disease","cell_type"])
adata.var_names= adata.var["gene_id"]
adata.obs["dataset_id"] = "QUERY"
adata.obs["dataset_title"] = "QUERY"


#census.close()
# %%
rfc = RandomForestClassifier()
rfc.fit(adata_census.obsm["scvi"], adata_census.obs["cell_type"].values)
adata.obs["predicted_cell_type"] = rfc.predict(adata.obsm["scvi"])

# let's get confidence scores
probabilities = rfc.predict_proba(adata.obsm["scvi"])

confidence = np.zeros(adata.n_obs)
for i in range(adata.n_obs):
    confidence[i] = probabilities[i][rfc.classes_ == adata.obs["predicted_cell_type"][i]]
    
confidence = np.max(probabilities, axis=1)

# Add confidence to adata and threshold prediction
adata.obs["confidence"] = confidence
adata.obs["predicted_cell_type"] = np.where(confidence >= 0.75, adata.obs["predicted_cell_type"], "unknown")
sc.pl.umap(adata, color=["dataset_title", "predicted_cell_type", "Cell_type"], ncols=1, na_in_legend=True, legend_fontsize=20)


# %%
