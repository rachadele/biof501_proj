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
    parser.add_argument('--tree_file', type=str, required=True, default="master_hierarchy.json")
    parser.add_argument('--query_path', type=str, default="Frontal_cortex_samples_from_C9-ALS,_C9-ALS_FTD_and_age_matched_control_brains.h5ad")
    parser.add_argument('--ref_paths', type=str, required=True) #nargs ="+")
    parser.add_argument('--ref_keys', type=str, nargs='+', default=["rachel_subclass", "rachel_class", "rachel_family"])
    parser.add_argument('--cutoff', type=int, default=0, help = "Cutoff threshold for positive classification")
    #parser.add_argument('--projPath', type=str, default=".")
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()


# Set organism and census_version from arguments
organism = args.organism
census_version = args.census_version
tree_file = args.tree_file
query_path=args.query_path
ref_path=args.ref_paths
ref_keys=args.ref_keys
cutoff=args.cutoff
# Read the JSON tree file
with open(args.tree_file, 'r') as file:
    tree = json.load(file)
    
query = ad.read_h5ad(query_path, backed="r")
query_name=os.path.basename(query_path).replace(".h5ad","")
#for ref_path in ref_paths:
ref=ad.read_h5ad(ref_path, backed="r")
ref_name=os.path.basename(ref_path).replace(".h5ad","")

probs = rfc_pred(ref=ref, query=query, ref_keys=ref_keys)
rocs= roc_analysis(probabilities=probs, 
                query=query, key=ref_keys[0])

outdir=os.path.join("roc",query_name, ref_name)
os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist

plot_roc_curves(metrics=rocs,
                title=f"{query_name} vs {ref_name}",
                save_path=os.path.join(outdir,"roc_results.png"))

#rocs_df = process_data(rocs)


query = classify_cells(query, ref_keys, cutoff=cutoff, probabilities=probs, tree=tree, threshold=False)
        
class_metrics = eval(query, 
                    ref_keys,
                    threshold=False)

class_metrics = update_classification_report(class_metrics) # replace 0 with nan for f1 with no support

for key in ref_keys:
    outdir=os.path.join("confusion",query_name, ref_name)
    plot_confusion_matrix(query_name, ref_name, key, class_metrics[key]["confusion"], output_dir=outdir)

#all_f1_scores={}
f1_data=[]
for key in ref_keys:
    classification_report = class_metrics[key]["classification_report"]
    # Extract F1 scores for each label
    for label, metrics in classification_report.items():
        if label not in ["macro avg","micro avg","weighted avg","accuracy"]:
        #   if isinstance(metrics, dict) and 'f1-score' in metrics:
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
                # 'accuracy': classification_report.get('accuracy', )
                })

    # Create DataFrame for the current key
df = pd.DataFrame(f1_data)
outdir ="f1_results"
os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
df.to_csv(os.path.join(outdir,f"{query_name}_{ref_name}_f1_scores.tsv"), sep="\t", index=False)



#outdir=os.path.join("umap",query_name,ref_name)
#os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist

#with plt.rc_context({'figure.figsize': (15, 15), 'savefig.dpi': 300}):  # Adjust size and dpi as needed                           
    ## Plot the UMAP
    #sc.pl.umap(
        #query, 
        #color=["predicted_" + key for key in ref_keys] + [key for key in ref_keys] + ["confidence"], 
        #ncols=2, na_in_legend=True, legend_fontsize=20, 
        #show=False  # Prevents immediate display, so we can save it with plt
    #)

    ## Save the figure using plt.savefig()
    #plt.savefig(
        #os.path.join(outdir,"umap.png"), 
        #dpi=300, 
        #bbox_inches='tight'
    #)

#plt.close()