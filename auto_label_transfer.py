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
from adata_functions import *
refs=adata_functions.get_census(organism="homo_sapiens", 
                                subsample=10, split_column="tissue", dims=20)

test_names=["Frontal cortex samples from C9-ALS, C9-ALS/FTD and age matched control brains"]
tests={}
for test_name in test_names:
    tests[test_name]=get_test_data(census_version=census_version, test_name=test_name, subsample=500, organism="homo_sapiens", split_key="dataset_title")
#tests = tests[~tests.obs['rachel_family'].isna(), :]

#if len(test_names) > 1:
  #  tests = split_anndata_by_obs(tests, obs_key="dataset_title")

join_key="observation_joinid"
relabel_path=os.path.join(projPath,"meta","relabel","gittings_relabel.tsv.gz")
 
queries={}                    
for test_name,test in tests.items():
   # relabel_test_path=os.path.join(projPath,"meta","relabel",test_name,"_relabel.tsv")
    test = relabel(test,relabel_path=relabel_path,
                        join_key=join_key,sep="\t")
  #  query= tests[test_name]
    queries[test_name] = process_query(test, projPath=projPath)




# %%
# Define the hierarchical structure as a dictionary with colnames and labels
tree = {
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
        "Vascular": {
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
# %%
#test
for test_name,query in queries.items():
    for ref_name,ref in refs.items():
    map_valid_labels(ref, query, tree, ref_keys)                                                                   

original_label = "Glutamatergic"

valid_label = get_valid_label(original_label, query_labels, tree)
print("Valid label found:", valid_label)


# %%
results = {}
for query_name, query in queries.items():
    for ref_name,ref in refs.items():
    results[query_name][ref_name] = classify_cells(adata_census=ref, query=query, ref_keys=ref_keys, tree= tree)
