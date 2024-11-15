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
from types import SimpleNamespace

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify cells given 1 ref and 1 query")
  #  parser.add_argument('--organism', type=str, default='homo_sapiens', help='Organism name (e.g., homo_sapiens)')
  #  parser.add_argument('--census_version', type=str, default='2024-07-01', help='Census version (e.g., 2024-07-01)')
    parser.add_argument('--tree_file', type=str, default="/space/grp/rschwartz/rschwartz/biof501_proj/meta/master_hierarchy.json")
    parser.add_argument('--query_path', type=str, default="/space/grp/rschwartz/rschwartz/biof501_proj/queries/Frontal_cortex_samples_from_C9-ALS,_C9-ALS_FTD_and_age_matched_control_brains.h5ad")
    parser.add_argument('--ref_paths', type=str, default="/space/grp/rschwartz/rschwartz/biof501_proj/refs/whole_cortex.h5ad") #nargs ="+")
    parser.add_argument('--ref_keys', type=str, nargs='+', default=["rachel_subclass", "rachel_class", "rachel_family"])
    parser.add_argument('--cutoff', type=float, default=0, help = "Cutoff threshold for positive classification")
    #parser.add_argument('--projPath', type=str, default=".")

    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

    
    
def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set variables from arguments
   # organism = args.organism
   # census_version = args.census_version
    tree_file = args.tree_file
    query_path = args.query_path
    ref_path = args.ref_paths
    ref_keys = args.ref_keys
    cutoff = args.cutoff

    # Read the JSON tree file
    with open(tree_file, 'r') as file:
        tree = json.load(file)

    # Load query and reference datasets
    query = ad.read_h5ad(query_path)
    #query=process_query(query, model_path="")
    query_name = os.path.basename(query_path).replace(".h5ad", "")
    ref = ad.read_h5ad(ref_path, backed="r")
    ref_name = os.path.basename(ref_path).replace(".h5ad", "")

    # Run classification and ROC analysis
    probs = rfc_pred(ref=ref, query=query, ref_keys=ref_keys)
    
    # eventually make this a data frame and save to disk, then make ROC another script
    
    probabilities = probs['rachel_subclass']['probabilities']
    class_labels = probs['rachel_subclass']['class_labels']

    # Create a DataFrame
    prob_df = pd.DataFrame(probabilities, columns=class_labels)
    #save data frame to inteim probs/ dir
    outdir=os.path.join("probs", query_name, ref_name)
    os.makedirs(outdir, exist_ok=True)
    prob_df.to_csv(os.path.join(outdir,"prob_df.tsv"),sep="\t")
 
    rocs = roc_analysis(probabilities=probs, query=query, key=ref_keys[0])

    outdir = os.path.join("roc", query_name, ref_name)
    os.makedirs(outdir, exist_ok=True)

    plot_roc_curves(metrics=rocs, title=f"{query_name} vs {ref_name}", save_path=os.path.join(outdir, "roc_results.png"))

    roc_df = process_roc(rocs, ref_name=ref_name, query_name=query_name)
    roc_df.to_csv(os.path.join(outdir,"roc_df.tsv"),sep="\t")
    
    ## eventually make this into a new script that takes probabilities matrix
    
    # Classify cells and evaluate
    query = classify_cells(query, ref_keys, cutoff=cutoff, probabilities=prob_df, tree=tree)
    class_metrics = eval(query, ref_keys)
    class_metrics = update_classification_report(class_metrics, ref_keys)

    # Plot confusion matrices
    for key in ref_keys:
        outdir = os.path.join("confusion", query_name, ref_name)
        plot_confusion_matrix(query_name, ref_name, key, class_metrics[key]["confusion"], output_dir=outdir)

    # Collect F1 scores
    f1_data = []
    for key in ref_keys:
        classification_report = class_metrics[key]["classification_report"]
        for label, metrics in classification_report.items():
            if label not in ["macro avg", "micro avg", "weighted avg", "accuracy"]:
                f1_data.append({
                    'query': query_name,
                    'reference': ref_name,
                    'label': label,
                    'f1_score': metrics['f1-score'],
                    'macro_f1': classification_report.get('macro avg', {}).get('f1-score', None),
                    'micro_f1': classification_report.get('micro avg', {}).get('f1-score', None),
                    'weighted_f1': classification_report.get('weighted avg', {}).get('f1-score', None),
                    'key': key,
                    'cutoff': cutoff
                })

    # Save F1 scores to a file
    df = pd.DataFrame(f1_data)
    outdir = "f1_results"
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{query_name}_{ref_name}_f1_scores.tsv"), sep="\t", index=False)

if __name__ == "__main__":
    main()
    
