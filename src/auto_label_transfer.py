#!/user/bin/python3



import subprocess
import importlib
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
import torch
from sklearn.ensemble import RandomForestClassifier
import adata_functions
from adata_functions import *
from pathlib import Path
current_directory = Path.cwd()
projPath = current_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import json
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
sc.set_figure_params(figsize=(10, 10), frameon=False)

# Read the JSON file
with open(os.path.join(projPath,"meta",'master_hierarchy.json'), 'r') as file:
    tree = json.load(file)


# Keys for harmonized labels at 3 levels of granularity
ref_keys = ["rachel_subclass","rachel_class","rachel_family"]
organism="homo_sapiens"
random.seed(1)
census_version="2024-07-01"
subsample=50
split_column="tissue"
dims=20
test_names=["Frontal cortex samples from C9-ALS, C9-ALS/FTD and age matched control brains"]



# Get model file link and download
model_file_path=setup(organism="homo_sapiens", version="2024-07-01")


refs=adata_functions.get_census(organism="homo_sapiens", 
                                subsample=10, split_column="tissue", dims=20)


tests={}
for test_name in test_names:
    tests[test_name]=get_test_data(census_version=census_version, test_name=test_name, subsample=500, organism="homo_sapiens", split_key="dataset_title")
#tests = tests[~tests.obs['rachel_family'].isna(), :]


queries = {}                    
for test_name,test in tests.items():
    test = relabel(test,relabel_path=os.path.join(projPath,"meta","relabel","gittings_relabel.tsv.gz"),
                        join_key="observation_joinid",sep="\t")
    queries[test_name] = process_query(test, model_file_path, projPath=projPath)



# %# Initialize defaultdict for thresholds and confusion
confusion = defaultdict(lambda: defaultdict(dict))
rocs = defaultdict(lambda: defaultdict(dict))
probs = defaultdict(lambda: defaultdict(dict))
class_metrics = defaultdict(lambda: defaultdict(dict))


for query_name, query in queries.items():

    for ref_name,ref in refs.items():
        all_probs = rfc_pred(ref=ref, query=query, ref_keys=ref_keys)
        probs[query_name][ref_name] = all_probs
        rocs[query_name][ref_name] = roc_analysis(probabilities=all_probs, 
                                                    query=query, key=ref_keys[0])
        new_query_name = query_name.replace(" ", "_").replace("/", "_")
        new_ref_name = ref_name.replace(" ", "_").replace("/", "_")
        outdir=os.path.join(projPath, "results","roc",new_query_name, new_ref_name)
        os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
        plot_roc_curves(metrics=rocs[query_name][ref_name],
                       title=f"{query_name} vs {ref_name}",
                        save_path=os.path.join(outdir,"roc_results.png"))
         



#threshold_df= process_data(rocs, var="optimal_threshold")
#average_thresholds = threshold_df.groupby('key')['optimal_threshold'].mean().to_dict()
## Example usage
#plot_distribution(threshold_df, projPath, var="optimal_threshold")

#auc_df= process_data(rocs, var="auc")
#average_auc = auc_df.groupby('key')['auc'].mean().to_dict()
## Example usage
#plot_distribution(auc_df, projPath, var="auc")


for query_name, query in queries.items():
    for ref_name in refs:
        probabilities = probs[query_name][ref_name]
        query = classify_cells(query, ref_keys, 0, probabilities, tree, threshold=False)
        
        class_metrics[query_name][ref_name] = eval(query, 
                                                    ref_keys,
                                                    threshold=False)
        new_query_name = query_name.replace(" ", "_").replace("/", "_")
        new_ref_name = ref_name.replace(" ", "_").replace("/", "_")                                                     
        # Plot the UMAP
        sc.pl.umap(
            query, 
            color=["confidence"] + ["predicted_" + key for key in ref_keys] + [key for key in ref_keys], 
            ncols=3, na_in_legend=True, legend_fontsize=20, 
            show=False  # Prevents immediate display, so we can save it with plt
        )
        outdir =os.path.join(projPath, "results", "umaps",new_query_name,new_ref_name)
        os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist

        # Save the figure using plt.savefig()
        plt.savefig(
            os.path.join(outdir, "umap.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
      
        plt.close()

 

all_f1_scores=combine_f1_scores(class_metrics, ref_keys) # Combine f1 scores into data frame
outdir =os.path.join(projPath, "results", "heatmaps")
os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
plot_f1_heatmaps(all_f1_scores, threshold=0, outpath=outdir, ref_keys=ref_keys)


for query_name in queries:
    for ref_name in refs:
            for key in ref_keys:
                 plot_confusion_matrix(query_name, ref_name, key,
                                      class_metrics[query_name][ref_name][key]["confusion"],
                                      output_dir=os.path.join(projPath,'results','confusion'))
 