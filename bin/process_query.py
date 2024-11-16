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
    parser.add_argument('--model_path', type=str, default="/space/grp/rschwartz/rschwartz/biof501_proj/scvi-human-2024-07-01", help='Path to the scvi model file')
 ##   parser.add_argument('--subsample_query', default=10, type=int)
  ##  parser.add_argument('--query_name', type=str, default="Frontal cortex samples from C9-ALS, C9-ALS/FTD and age matched control brains")
    parser.add_argument('--relabel_path', type=str, default="/space/grp/rschwartz/rschwartz/biof501_proj/meta/gittings_relabel.tsv.gz")
    parser.add_argument('--query_path', type=str, default="/space/grp/rschwartz/rschwartz/biof501_proj/queries/Frontal_cortex_samples_from_C9-ALS,_C9-ALS_FTD_and_age_matched_control_brains.h5ad")
    parser.add_argument('--batch_key', type=str, default="sample")
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def main():
  # Parse command line arguments
  args = parse_arguments()


  # Set organism and census_version from arguments

  model_path = args.model_path
  #tree_file = args.tree_file
  ##subsample_query = args.subsample_query
  ##test_name=args.test_name
  query_path =args.query_path
  relabel_path=args.relabel_path
  batch_key=args.batch_key
  
  query = ad.read_h5ad(query_path)
  query = relabel(query,relabel_path=relabel_path, join_key="observation_joinid",sep="\t")
  query = process_query(query, model_path, batch_key)
  new_query_name = os.path.basename(query_path).replace(".h5ad","_processed")
  query.write_h5ad(f"{new_query_name}.h5ad")
  
if __name__ == "__main__":
    main()