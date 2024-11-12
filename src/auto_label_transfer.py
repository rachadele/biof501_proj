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
#import glob
import cellxgene_census
#import leidenalg
#import igraph
import scvi
#import scipy
from scipy.sparse import csr_matrix
import warnings
import cellxgene_census
import cellxgene_census.experimental
import scvi
#import tempfile
#import botocore
import torch
#import gzip
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

# Read the JSON file
with open(os.path.join(projPath,"meta",'master_hierarchy.json'), 'r') as file:
    tree = json.load(file)

# Set pandas to display all columns
pd.set_option('display.max_columns', None)
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
sc.set_figure_params(figsize=(10, 10), frameon=False)
#print("Last run with scvi-tools version:", scvi.__version__)
#sys.path.append('/app') 
organism="homo_sapiens"
# Keys for harmonized labels at 3 levels of granularity
ref_keys = ["rachel_subclass","rachel_class","rachel_family"]

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
setup()
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
from collections import defaultdict
importlib.reload(adata_functions)
from adata_functions import *

# Initialize defaultdict for thresholds and confusion
confusion = defaultdict(lambda: defaultdict(dict))
rocs = defaultdict(lambda: defaultdict(dict))
probs = defaultdict(lambda: defaultdict(dict))
class_metrics = defaultdict(lambda: defaultdict(dict))
for query_name, query in queries.items():
    #probs[query_name] = {}
  #  [query_name] = {}
    for ref_name,ref in refs.items():
        all_probs = rfc_pred(ref=ref, query=query, ref_keys=ref_keys)
        probs[query_name][ref_name] = all_probs
        
        print(query_name)
        print(ref_name)
     #  for key in ref_keys:
            
        print(check_column_ties(all_probs["rachel_subclass"]["probabilities"],all_probs["rachel_subclass"]["class_labels"]))
        rocs[query_name][ref_name] = roc_analysis(probabilities=all_probs, 
                                                    query=query, key=ref_keys[0])
        new_query_name = query_name.replace(" ", "_").replace("/", "_")
        new_ref_name = ref_name.replace(" ", "_").replace("/", "_")
        outdir=os.path.join(projPath, "results","roc",new_query_name, new_ref_name)
        os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
        plot_roc_curves(metrics=rocs[query_name][ref_name],
                       title=f"{query_name} vs {ref_name}",
                        save_path=os.path.join(outdir,"roc_results.png"))
         

# %%
threshold_df= process_data(rocs, var="optimal_threshold")
average_thresholds = threshold_df.groupby('key')['optimal_threshold'].mean().to_dict()
# Example usage
plot_distribution(threshold_df, projPath, var="optimal_threshold")

auc_df= process_data(rocs, var="auc")
average_auc = auc_df.groupby('key')['auc'].mean().to_dict()
# Example usage
plot_distribution(auc_df, projPath, var="auc")
# %%

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

 
# %%  

all_f1_scores=combine_f1_scores(class_metrics, ref_keys) # Combine f1 scores into data frame
outdir =os.path.join(projPath, "results", "heatmaps")
os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
plot_f1_heatmaps(all_f1_scores, threshold=0, outpath=outdir, ref_keys=ref_keys)

# %%
for query_name in queries:
    for ref_name in refs:
            for key in ref_keys:
                 plot_confusion_matrix(query_name, ref_name, key,
                                      class_metrics[query_name][ref_name][key]["confusion"],
                                      output_dir=os.path.join(projPath,'results', "no_thresholds",'confusion'))
 