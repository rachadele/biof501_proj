#!/user/bin/python3

from pathlib import Path
import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import cellxgene_census
import scvi
from scipy.sparse import csr_matrix
import warnings
import cellxgene_census
import cellxgene_census.experimental
import scvi
from sklearn.ensemble import RandomForestClassifier
import adata_functions
from adata_functions import *
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
import json

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--organism', type=str, default='homo_sapiens', help='Organism name (e.g., homo_sapiens)')
    parser.add_argument('--census_version', type=str, default='2024-07-01', help='Census version (e.g., 2024-07-01)')
    parser.add_argument('--subsample_ref', type=int, default=10)
    parser.add_argument('--relabel_path', type=str, default="/biof501_proj/meta/relabel/census_map_human.tsv")
    
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()


# Set organism and census_version from arguments
organism = args.organism
census_version = args.census_version
#model_file_path = args.model_file_path
#tree_file = args.tree_file
subsample_ref = args.subsample_ref
#subsample_query = args.subsample_query
relabel_path = args.relabel_path

# Read the JSON tree file
#with open(args.tree_file, 'r') as file:
   # tree = json.load(file)


refs=adata_functions.get_census(organism=organism, 
                                subsample=subsample_ref, split_column="tissue", dims=20, relabel_path=relabel_path)

print("finished fetching anndata")
outdir="refs"
os.makedirs(outdir, exist_ok=True) 

for ref_name, ref in refs.items():
    new_ref_name = ref_name.replace(" ", "_").replace("\\/", "_")
 # Create the directory if it doesn't exist
    ref.write(os.path.join(outdir,f"{new_ref_name}.h5ad"))
    
