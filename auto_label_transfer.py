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
import scipy
from scipy.sparse import csr_matrix
import warnings
import cellxgene_census
import cellxgene_census.experimental
import scvi
import tempfile
import botocore
import torch
import gzip
from sklearn.ensemble import RandomForestClassifier
import adata_functions
from adata_functions import *
projPath = "."
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import yaml
import anytree
from anytree import Node, RenderTree
from anytree.importer import DictImporter




# Set pandas to display all columns
pd.set_option('display.max_columns', None)
#scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
sc.set_figure_params(figsize=(10, 10), frameon=False)
#print("Last run with scvi-tools version:", scvi.__version__)
sys.path.append('/app') 
organism="homo_sapiens"
# Keys for harmonized labels at 3 levels of granularity
ref_keys = ["rachel_family","rachel_class","rachel_subclass"]

random.seed(1)
census_version="2024-07-01"
organism="homo_sapiens"
subsample=50
split_column="tissue"
dims=20

# %%
def setup(organism="homo_sapiens", version="2024-07-01"):
    organism=organism.replace(" ", "_") 
    census = cellxgene_census.open_soma(census_version=version)
    outdir = f"{organism}-{version}"  # Concatenate strings using f-string
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Check if the model file exists
    model_file_path = os.path.join(outdir, "model.pt")
    if not os.path.exists(model_file_path):
        # Get scVI model info
        scvi_info = cellxgene_census.experimental.get_embedding_metadata_by_name(
            embedding_name="scvi",
            organism=organism,
            census_version=version,
        )

        # Extract the model link
        model_link = scvi_info["model_link"]
        date = model_link.split("/")[5]
        url = os.path.join("https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/", date, organism, "model.pt")

        # Download the model using wget if it doesn't already exist
        subprocess.run(["wget", "--no-check-certificate", "-q", "-O", model_file_path, url])
    else:
        print(f"File already exists at {model_file_path}, skipping download.")

    return(model_file_path)


# %%
# Set random seed for reproducibility of subsampling
# Should I pass this to individual functions?
importlib.reload(adata_functions)
census = cellxgene_census.open_soma(census_version=census_version)

refs=adata_functions.get_census(organism="homo_sapiens", 
                                census=census, subsample=50, split_column="tissue", dims=20)

test_names=["Cortical brain samples from C9-ALS, C9-ALS/FTD, C9-FTD patients and age matched controls"]
tests=get_test_data(census_version=version, test_names=test_names, subsample=500)
tests = split_anndata_by_obs(tests, "dataset_title")

for test_name in test_names():
    relabel_test_path=os.path.join(projPath,"meta","relabel",test_name,"_relabel.tsv")
    query= tests[test_name]
    adata_query = process_adata_query(adata_query, projPath=projPath)



# %%
# Specify your YAML file path here
from anytree.importer import DictImporter
yaml_filepath = os.path.join(projPath,"meta","master_hierarchy.yaml")  # Update to your actual file path

with open(yaml_filepath, 'r') as file:
    dct = yaml.safe_load(file)
tree = DictImporter().import_(dct)
print(RenderTree(tree))

# %%

for ref in refs:
    classify_cells(adata_census=ref, adata_query=adata_query, ref_keys=ref_keys, tree= tree)
