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
    #parser.add_argument('--tree_file', type=str, required=True, help='Path to the tree JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the scvi model file')
    parser.add_argument('--subsample_query', default=50, type=int)
    #parser.add_argument('--projPath', type=str, default=".")
    parser.add_argument('--test_name', type=str, default="Frontal cortex samples from C9-ALS, C9-ALS/FTD and age matched control brains")
    parser.add_argument('--relabel_path', type=str, default="../meta/relabel/gittings_relabel.tsv.gz")
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()


# Set organism and census_version from arguments
organism = args.organism
census_version = args.census_version
model_path = args.model_path
#tree_file = args.tree_file
subsample_query = args.subsample_query
test_name=args.test_name
relabel_path=args.relabel_path
# Read the JSON tree file
#with open(args.tree_file, 'r') as file:
   # tree = json.load(file)
    
#for query_name in test_names:
query=get_test_data(census_version=census_version, test_name=test_name, 
                                subsample=subsample_query, organism=organism, split_key="dataset_title")
query = relabel(query,relabel_path=relabel_path,
join_key="observation_joinid",sep="\t")
query = process_query(query, model_path)
new_query_name = test_name.replace(" ", "_").replace("/", "_").replace("(","").replace(")","")
#outdir=os.path.join("h5ad","queries")
#os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
query.write(f"{new_query_name}.h5ad")