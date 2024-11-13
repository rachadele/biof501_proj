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
import ast

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--ref_keys', type=str, nargs='+', default=["rachel_subclass", "rachel_class", "rachel_family"])
    parser.add_argument('--cutoff', type=int, default=0, help = "Cutoff used for classification")
    parser.add_argument('--f1_results', type=str, required=True, nargs='+')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()


# Set organism and census_version from arguments
ref_keys=args.ref_keys
f1_results = args.f1_results#.replace("[", "").replace("]", "").split(",")
cutoff=args.cutoff

all_f1_scores = {}
f1_scores=pd.DataFrame()

for file in f1_results:
    # Read each file into a DataFrame
    x = pd.read_csv(file, sep="\t")
    
    # Get the unique keys from the 'key' column
    keys = x["key"].unique()
    
    # Loop through each unique key
    for key in keys:
        # Filter rows where the 'key' matches
        subset = x[x["key"] == key]
        
        # If this key doesn't exist in the dictionary, initialize an empty DataFrame
        if key not in all_f1_scores:
            all_f1_scores[key] = subset
        else:
            # If the key already exists, concatenate the new subset to the existing DataFrame
            all_f1_scores[key] = pd.concat([all_f1_scores[key], subset], ignore_index=True)

    

plot_f1_heatmaps(all_f1_scores, threshold=cutoff, outpath="f1_plots", ref_keys=ref_keys)

