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
import matplotlib.pyplot as plt

# Set pandas to display all columns
pd.set_option('display.max_columns', None)
scvi.settings.seed = 0
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

refs=adata_functions.get_census(organism="homo_sapiens", 
                                subsample=10, split_column="tissue", dims=20)

test_names=["Cortical brain samples from C9-ALS, C9-ALS/FTD, C9-FTD patients and age matched controls"]
tests=get_test_data(census_version=cemsus_version, test_names=test_names, subsample=500)
tests = split_anndata_by_obs(tests, "dataset_title")

for test_name in test_names():
    relabel_test_path=os.path.join(projPath,"meta","relabel",test_name,"_relabel.tsv")
    query= tests[test_name]
    adata_query = process_adata_query(adata_query, projPath=projPath)



# %%
# Specify your YAML file path here
yaml_filepath = os.path.join(projPath,"meta","master_hierarchy.yaml")  # Update to your actual file path

with open(yaml_filepath, 'r') as file:
    dct = yaml.safe_load(file)


# %%
# Define the hierarchical structure as a dictionary with colnames and labels
cell_hierarchy = {
    "GABAergic": {
        "colname": "rachel_family",
        "CGE": {
            "colname": "rachel_class",
            "LAMP5": {
                "colname": "rachel_subclass"
            },
            "VIP": {
                "colname": "rachel_subclass"
            },
            "SNCG": {
                "colname": "rachel_subclass"
            }
        },
        "MGE": {
            "colname": "rachel_class",
            "PVALB": {
                "colname": "rachel_subclass"
            },
            "SST": {
                "colname": "rachel_subclass"
            },
            "Chandelier": {
                "colname": "rachel_subclass"
            }
        }
    },
    "Glutamatergic": {
        "colname": "rachel_family",
        "L2/3-6 IT": {
            "colname": "rachel_class"
        },
        "deep layer non-IT": {
            "colname": "rachel_class",
            "L5 ET": {
                "colname": "rachel_subclass"
            },
            "L5/6 NP": {
                "colname": "rachel_subclass"
            },
            "L6 CT": {
                "colname": "rachel_subclass"
            },
            "L6b": {
                "colname": "rachel_subclass"
            }
        }
    },
    "Non-neuron": {
        "colname": "rachel_family",
        "Oligodendrocyte lineage": {
            "colname": "rachel_class",
            "Oligodendrocyte": {
                "colname": "rachel_subclass"
            },
            "OPC": {
                "colname": "rachel_subclass"
            }
        },
        "Astrocyte": {
            "colname": "rachel_class"
        },
        "Immune/Vasculature": {
            "colname": "rachel_class",
            "Pericyte": {
                "colname": "rachel_subclass"
            },
            "VLMC": {
                "colname": "rachel_subclass"
            },
            "Endothelial": {
                "colname": "rachel_subclass"
            }
        }
    }
}

# Define a recursive function to find a valid label
def find_valid_label(hierarchy, current_label):
    if current_label in hierarchy:
        # If the current label exists, return it
        return hierarchy[current_label]
    
    # Check for parent if the current label is not found at this level
    for key, value in hierarchy.items():
        if isinstance(value, dict):
            # Recurse deeper if there's a nested structure
            result = find_valid_label(value, current_label)
            if result:
                return result
    return None  # If no valid label is found

# Example usage
current_label = "LAMP5"
valid_label = find_valid_label(cell_hierarchy, current_label)

if valid_label:
    print(f"Valid label for {current_label}: {valid_label}")
else:
    print(f"No valid label found for {current_label}")

for ref in refs:
    classify_cells(adata_census=ref, adata_query=adata_query, ref_keys=ref_keys, tree= tree)


# Example usage
current_label = "LAMP5"
valid_label = find_valid_label(cell_hierarchy, current_label)

if valid_label:
    print(f"Valid label for {current_label}: {valid_label}")
else:
    print(f"No valid label found for {current_label}")