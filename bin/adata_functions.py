import pandas as pd
import numpy as np
import scanpy as sc
import random
import cellxgene_census
import cellxgene_census.experimental
import os
import anndata as ad
import scvi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
scvi.settings.seed = 0
from pathlib import Path
#current_directory = Path.cwd()
projPath = "/space/grp/rschwartz/rschwartz/biof501_proj/bin"

import subprocess


def setup(organism="homo_sapiens", version="2024-07-01"):
    organism=organism.replace(" ", "_") 
    #census = cellxgene_census.open_soma(census_version=version)
    outdir = f"scvi-human-{version}"  # Concatenate strings using f-string
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Check if the model file exists
    model_file_path = os.path.join(outdir, "model.pt")
    #if not os.path.exists(model_file_path):
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
# else:
     #   print(f"File already exists at {model_file_path}, skipping download.")

    return(outdir)

# Subsample x cells from each cell type if there are n>x cells present
#ensures equal representation of cell types in reference
def subsample_cells(data, filtered_ids, subsample=500):
    # Filter data based on filtered_ids
    obs = data[data['soma_joinid'].isin(filtered_ids)]
    celltypes = obs['cell_type'].unique()
    final_idx = []
    for celltype in celltypes:
        celltype_ids = obs[obs['cell_type'] == celltype]['soma_joinid'].values
        # Sample if there are enough observations, otherwise take all
        if len(celltype_ids) >= subsample:
            subsampled_cell_idx = random.sample(list(celltype_ids), subsample)
        else:
            subsampled_cell_idx = celltype_ids.tolist()
        # Append subsampled indices to final list
        final_idx.extend(subsampled_cell_idx)

    # Return final indices
    return final_idx

def relabel(adata, relabel_path, join_key, sep="\t"):
    # Read the relabel table from the file
    relabel_df = pd.read_csv(relabel_path, sep=sep)  # Adjust the separator as needed
    # Take the first column as the join key
   # join_key = relabel_df.columns[0]
    # Ensure the join_key is in both the AnnData object and the relabel DataFrame
    if join_key not in adata.obs.columns:
        raise ValueError(f"{join_key} not found in AnnData object observations.")
    if join_key not in relabel_df.columns:
        raise ValueError(f"{join_key} not found in relabel DataFrame.")
    # Perform the left join to update the metadata
    adata.obs = adata.obs.merge(relabel_df, on=join_key, how='left', suffixes=(None, "_y"))
    columns_to_drop = [col for col in adata.obs.columns if col.endswith('_y')]
    adata.obs.drop(columns=columns_to_drop, inplace=True)
    return adata


def extract_data(data, filtered_ids, subsample=10, organism=None, census=None, 
    obs_filter=None, cell_columns=None, dataset_info=None, dims=20, relabel_path="/biof501_proj/meta/relabel/census_map_human.tsv'"):
    
    brain_cell_subsampled_ids = subsample_cells(data, filtered_ids, subsample)
    # Assuming get_seurat is defined to return an AnnData object
    adata = cellxgene_census.get_anndata(
        census=census,
        organism=organism,
        obs_value_filter=obs_filter,  # Ensure this is constructed correctly
        obs_column_names=cell_columns,
        obs_coords=brain_cell_subsampled_ids,
        var_value_filter = "nnz > 50",
        obs_embeddings=["scvi"])
    sc.pp.filter_genes(adata, min_cells=3) 
    print("Subsampling successful.")
    # Filter out genes that are not expressed in at least 3 cells
    #adata = adata[adata.X.sum(axis=0) >= 3, :]
    # Preprocessing
    #sc.pp.normalize_total(adata)
    #sc.pp.scale(adata)
    #sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    #sc.tl.pca(adata, n_comps=dims)
    sc.pp.neighbors(adata, use_rep="scvi",n_neighbors=30)
    #sc.tl.leiden(query)
    sc.tl.umap(adata)
    # Merging metadata with dataset_info
    newmeta = adata.obs.merge(dataset_info, on="dataset_id", suffixes=(None,"y"))
    adata.obs = newmeta
    # Assuming relabel_wrapper is defined
    adata = relabel(adata, relabel_path=relabel_path, join_key="cell_type", sep='\t')
    # Convert all columns in adata.obs to factors (categorical type in pandas)
    return adata

def split_and_extract_data(data, split_column, subsample=500, organism=None, census=None, cell_columns=None, dataset_info=None, dims=20, relabel_path="/biof501_proj/meta/relabel/census_map_human.tsv"):
    # Get unique split values from the specified column
    unique_values = data[split_column].unique()
    refs = {}

    for split_value in unique_values:
        # Filter the brain observations based on the split value
        filtered_ids = data[data[split_column] == split_value]['soma_joinid'].values
        obs_filter = f"{split_column} == '{split_value}'"
        
        adata = extract_data(data, filtered_ids, subsample, organism, census, obs_filter, 
                             cell_columns, dataset_info, dims=dims, relabel_path=relabel_path)
        dataset_titles = adata.obs['dataset_title'].unique()

        if len(dataset_titles) > 1:
            name_to_use = split_value
        else:
            name_to_use = dataset_titles[0]

        refs[name_to_use] = adata

    return refs

def get_census(census_version="2024-07-01", organism="homo_sapiens", subsample=10, split_column="tissue", dims=20, 
               ref_collections=["Transcriptomic cytoarchitecture reveals principles of human neocortex organization"],
               relabel_path="/biof501_proj/meta/relabel/census_map_human.tsv"):

    census = cellxgene_census.open_soma(census_version=census_version)
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = cellxgene_census.get_obs(census, organism,
        value_filter=(
            "tissue_general == 'brain' and "
            "is_primary_data == True and "
            "disease == 'normal' "
        ))
    
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id", suffixes=(None,"_y"))
    brain_obs.drop(columns=['soma_joinid_y'], inplace=True)
    brain_obs_filtered = brain_obs
    # Filter based on organism
    if organism == "homo_sapiens":
        brain_obs_filtered = brain_obs[
            brain_obs['collection_name'].isin(ref_collections)]
        brain_obs_filtered = brain_obs_filtered[~brain_obs_filtered['cell_type'].isin(["unknown", "glutamatergic neuron"])]
    elif organism == "mus_musculus":
        brain_obs_filtered = brain_obs[
            brain_obs['collection_name'].isin(ref_collections) 
        ]
    else:
       raise ValueError("Unsupported organism")

    # Adjust organism naming for compatibility
    organism_name_mapping = {
        "homo_sapiens": "Homo sapiens",
        "mus_musculus": "Mus musculus"
    }
    organism = organism_name_mapping.get(organism, organism)

    cell_columns = [
        "assay", "cell_type", "tissue",
        "tissue_general", "suspension_type",
        "disease", "dataset_id", "development_stage",
        "soma_joinid"
    ]
    # Get individual datasets and embeddings
    refs = split_and_extract_data(
        brain_obs_filtered, split_column=split_column,
        subsample=subsample, organism=organism,
        census=census, cell_columns=cell_columns,
        dataset_info=dataset_info, dims=dims,
        relabel_path=relabel_path
    )
    # Get embeddings for all data together
    filtered_ids = brain_obs_filtered['soma_joinid'].values
    adata = extract_data(
        brain_obs_filtered, filtered_ids,
        subsample=subsample, organism=organism,
        census=census, obs_filter=None,
        cell_columns=cell_columns, dataset_info=dataset_info, dims=dims,
        relabel_path=relabel_path
    )
    refs["whole cortex"] = adata

    for name, ref in refs.items():
        dataset_title = name.replace(" ", "_")
        for col in ref.obs.columns:
            if ref.obs[col].dtype.name =='category':
    # Convert to Categorical and remove unused categories
                ref.obs[col] = pd.Categorical(ref.obs[col].cat.remove_unused_categories())
    
    return refs



def process_query(query, model_file_path, batch_key="sample"):
    # Ensure the input AnnData object is valid
    if not isinstance(query, ad.AnnData):
        raise ValueError("Input must be an AnnData object.")

    # Assign ensembl_id to var
    #query.var["ensembl_id"] = query.var["feature_id"]
    if "feature_id" in query.var.columns:
        query.var.set_index("feature_id", inplace=True)

    query.obs["n_counts"] = query.X.sum(axis=1)
    query.obs["joinid"] = list(range(query.n_obs))
    query.obs["batch"] = query.obs[batch_key]

    # Filter out missing HGNC features
    query = query[:, query.var["feature_name"].notnull().values].copy()

    # Prepare the query AnnData for scVI
    scvi.model.SCVI.prepare_query_anndata(query, model_file_path)
    vae_q = scvi.model.SCVI.load_query_data(query, model_file_path)

    # Set the model to trained and get latent representation
    vae_q.is_trained = True
    latent = vae_q.get_latent_representation()
    query.obsm["scvi"] = latent

    # Add tissue information and dataset identifiers
    #query.obs["tissue"] = tissue
    #query.obs["tissue_general"] = "brain"
    #query.var_names = query.var["gene_id"]
    #query.obs["dataset_id"] = dataset_id
    #query.obs["dataset_title"] = dataset_title

    # Compute neighbors and UMAP
    sc.pp.neighbors(query, n_neighbors=30, use_rep="scvi")
    sc.tl.umap(query)
    sc.tl.leiden(query)

    return query


# Function to find a node's parent in the tree
def find_parent_label(tree, target_label, current_path=None):
    if current_path is None:
        current_path = []
    for key, value in tree.items():
        # Add the current node to the path
        current_path.append(key)
        # If we found the target, return the parent label if it exists
        if key == target_label:
            if len(current_path) > 1:
                return current_path[-2]  # Return the parent label
            else:
                return None  # No parent if we're at the root
        # Recurse into nested dictionaries if present
        if isinstance(value, dict):
       #     print(value)
            result = find_parent_label(value, target_label, current_path)
           # print(result)
            if result:
                return result
        # Remove the current node from the path after processing
        current_path.pop()
    return None

# Recursive function to get the closest valid label
def get_valid_label(original_label, query_labels, tree):
    # Base case: if the label exists in query, return it
    if original_label in query_labels:
        return original_label
    # Find the parent label in the tree
    parent_label = find_parent_label(tree, original_label)
    # Recursively check the parent label if it exists
    if parent_label:
        return get_valid_label(parent_label, query_labels, tree)
    else:
        return None  # Return None if no valid parent found

# Example usage

def map_valid_labels(ref, query, tree, ref_keys):
    # make sure ref and query have the same levels
 
    query_labels=pd.concat([query.obs[key] for key in ref_keys]).unique()
    ref_labels = pd.concat([ref.obs[key] for key in ref_keys]).unique()
    #query.obs[ref_keys]
    #ref.obs[ref_keys]
    # if reference label is not in any of the query labels at any level, replace with parent
    # this accounts for queries without detailed subtypes
    # e.g. lau: can only evaluate at the "rachel_class" level
    for ref_label in ref_labels:
        new_label = get_valid_label(ref_label, query_labels, tree)
        # Replace all instances of ref_label with new_label in ref
      #  if new_label:
        for key in ref_keys:
                # Replace ref_label with new_label in ref
            ref.obs[key] = ref.obs[key].replace(ref_label, new_label)
    
    # if query label not in reference, don't evaluate
    # e.g. "Chandelier" is not in some of the references, and will likely be
    # annotated as some other GABAergic type
    # what do in this situation?
    # change to "GABAergic" in query?
    # this likely won't help, since other valid subclass labels already exist in the query
    # best solution is to not evaluate
    # doesn't account for "unknowns" ?
    # have to handle this case
    for query_label in query_labels:
        if query_label not in ref_labels:
            query.obs[key] = query.obs[key].replace(query_label, None)
    
    return ref,query

from sklearn.ensemble import RandomForestClassifier
import numpy as np


def find_node(tree, target_key):
    """
    Recursively search the tree for the target_key and return the corresponding node. 
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            if key == target_key:  # If we've found the class at this level
                return value  # Return the current node
            else:
                # Recurse deeper into the tree
                result = find_node(value, target_key)
                if result:
                    return result
    return None  # Return None if the target key is not found


# Helper function to recursively gather all subclasses under a given level
def get_subclasses(node, colname):
    subclasses = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and value.get("colname") == colname:
                subclasses.append(key)
            else:
                subclasses.extend(get_subclasses(value, colname))
    return subclasses


def rfc_pred(ref, query, ref_keys):
    """
    Fit a RandomForestClassifier at the most granular level and aggregate probabilities for higher levels.
    
    Parameters:
    - ref: Reference data with labels.
    - query: Query data for prediction.
    - ref_keys: List of ordered keys from most granular to highest level (e.g., ["rachel_subclass", "rachel_class", "rachel_family"]).
    - tree: Dictionary representing the hierarchy of classes.
    
    Returns:
    - probabilities: Dictionary with probabilities for each level of the hierarchy.
    """
    probabilities = {}
    
    # The most granular key is the first in the ordered list
    granular_key = ref_keys[0]
    
    # Initialize and fit the RandomForestClassifier at the most granular level
    rfc = RandomForestClassifier(class_weight='balanced', random_state=42, max_depth=20, )
    rfc.fit(ref.obsm["scvi"], ref.obs[granular_key].values)
    # Predict probabilities at e most granular level
    probs_granular = rfc.predict_proba(query.obsm["scvi"])
    class_labels_granular = rfc.classes_
    base_score = rfc.score(query.obsm["scvi"], query.obs[granular_key].values)

    # Store granular level probabilities
    probabilities[granular_key] = {
        "probabilities": probs_granular,
        "class_labels": class_labels_granular,
        "accuracy": base_score
    }
    
    return probabilities 



def roc_analysis(probabilities, query, key, specified_threshold=None):
    optimal_thresholds = {}
    metrics={}
  #  for key in ref_keys:
       # print(key) 
    probs = probabilities[key]["probabilities"]
    class_labels = probabilities[key]["class_labels"]
    optimal_thresholds[key] = {}
        
    # Binarize the class labels for multiclass ROC computation
    true_labels = label_binarize(query.obs[key].values, classes=class_labels)
        
    # Find the optimal threshold for each class
    metrics[key] = {}
    for i, class_label in enumerate(class_labels):
        optimal_thresholds[key][class_label] = {}
        # check for positive samples
        # usually positive samples are 0 when a ref label is
        # replaced with a parent label
        # since it is not in the original query labels
        # but it is being annotated during the label transfer
        # these should not be evaluated ?
        # incorrect classifications will be reflected in the AUC and F1 of the og label
        # eg. ET is not in query so it is changed to "deep layer non-IT"
        # but these cells are CT or NP in the ref, so they're still incorrect
        # not entirely sure how to deal with this
        positive_samples = np.sum(true_labels[:, i] == 1)
        if positive_samples == 0:
            print(f"Warning: No positive samples for class {class_label}, skipping eval and setting threshold to 0.5")
            optimal_thresholds[key][class_label] = 0.5
        elif positive_samples > 0:
            metrics[key][class_label]={}
            # True label one hot encoding at class label index = 
            # vector of all cells which are either 1 = label or 0 = not label
            # probs = probability vector for all cells given class label
            fpr, tpr, thresholds = roc_curve(true_labels[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)                
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            if optimal_threshold == float('inf'):
                optimal_threshold = 0 
            optimal_thresholds[key][class_label]=optimal_threshold
            metrics[key][class_label]["tpr"] = tpr
            metrics[key][class_label]["fpr"] = fpr
            metrics[key][class_label]["auc"] = roc_auc
            metrics[key][class_label]["optimal_threshold"] = optimal_threshold

    return metrics


def process_data(rocs): 
    # Populate the list with threshold data
    data = []

    for key, results in rocs.items():
        for ref, ref_data in query_data.items():
            for key, roc in rocs.items():
                if roc:
                    for class_label, class_data in roc.items():
                        if class_data:
                            data.append({
                                "ref": ref,
                                "query": query,
                                "key": key, 
                                "label": class_label, 
                                "roc": class_data["auc"],
                                "threshold": class_data["optimal_threshold"]
                              #   f'{var}': class_data[var]
                            })

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    return df

def plot_distribution(df, projPath, var):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='label', y=var, palette="Set2")
    plt.xlabel('Key', fontsize=14)
    plt.ylabel(f"{var}", fontsize=14)
    plt.title(f'Distribution of {var} across all References', fontsize=30)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_path = os.path.join(projPath, "results", f"{var}_distribution.png")
    plt.savefig(save_path)


def check_column_ties(probabilities, class_labels):
    """
    Checks for column ties (multiple columns with the same maximum value) in each row.

    Parameters:
    - probabilities (numpy.ndarray): 2D array where rows represent samples and columns represent classes.
    - class_labels (list): List of class labels corresponding to the columns.

    Returns:
    - tie_rows (list): List of row indices where ties occur.
    - tie_columns (dict): Dictionary where the key is the row index and the value is a list of tied column indices and their class labels.
    """
    # Find the maximum probability for each row
    max_probs = np.max(probabilities, axis=1)

    # Check if there are ties (multiple columns with the maximum value in the row)
    ties = np.sum(probabilities == max_probs[:, None], axis=1) > 1

    # Get the indices of rows with ties
    tie_rows = np.where(ties)[0]

    # Find the columns where the tie occurs and associated class labels
    tie_columns = {}
    for row in tie_rows:
        tied_columns = np.where(probabilities[row] == max_probs[row])[0]
        tie_columns[row] = [(col, class_labels[col]) for col in tied_columns]
    
    return tie_rows, tie_columns

def classify_cells(query, ref_keys, cutoff, probabilities, tree, **kwargs):
    threshold = kwargs.get('threshold', True)  # Or some other default value
    class_metrics = {}
 #   for key in ref_keys:  
    key = ref_keys[0]
    class_metrics[key]={}
    class_labels = probabilities[key]["class_labels"]
    predicted_classes = []
        # Convert thresholds to a numpy array for faster comparison
    #threshold_array = np.array(cutoff * query.n_obs)
    # Vectorized probability retrieval and decision-making
    class_probs = np.array([probabilities[key]["probabilities"][i] for i in range(query.n_obs)])  # Shape: (query.n_obs, num_classes)
    class_labels = np.array(probabilities[key]["class_labels"])  # Shape: (num_classes,)
    # Use np.argmax to find the class with the highest probability
    if threshold:
        # Find the class with the maximum probability for each cell
        max_class_indices = np.argmax(class_probs, axis=1) # shape 500,
        # index of class
        max_class_probs = np.max(class_probs, axis=1) # shape 500,
        #max probabilitiy
        # need to break ties somehow here so that levels agree
        # fml
            
        # Set predicted classes to "unknown" if the max probability does not meet the threshold
        predicted_classes = [
            class_labels[i] if prob > cutoff else "unknown"
            for cell, (i, prob) in enumerate(zip(max_class_indices, max_class_probs))
        ]  # i = class index
            # prob = prob
            # cell = cell index
    else:
        # Direct prediction without threshold filtering
        predicted_classes = class_labels[np.argmax(class_probs, axis=1)]          
            
    # Store predictions and confidence in `query`
    query.obs["predicted_" + key] = predicted_classes
    query.obs["confidence"] = np.max(probabilities[key]["probabilities"], axis=1)
    
    query = aggregate_preds(query, ref_keys, tree)
    return query

def aggregate_preds(query, ref_keys, tree):
    
    preds = np.array(query.obs["predicted_" + ref_keys[0]])
    query.obs.index = query.obs.index.astype(int)
  #  for label in pred_levels:
   #     find_node
    for higher_level_key in ref_keys[1:]: 
        query.obs["predicted_" + higher_level_key] = "unknown"  # Initialize to account for unknowns preds
        # Skip the first (granular) level
        ## Get all possible classes for this level (e.g. "GABAergic", "Glutamatergic", "Non-neuron")
        subclasses = get_subclasses(tree, higher_level_key) 
        
        for higher_class in subclasses: # eg "GABAergic"
            node = find_node(tree, higher_class) # find position in tree dict
            valid = get_subclasses(node, ref_keys[0]) # get all granular labels falling under this class
            ## eg all GABAergic subclasses
            if not valid:
                print("no valid subclasses")
                continue  # Skip if no subclasses found   

            # Get the indices of cells in `preds` that match any of the valid subclasses
            cells_to_agg = np.where(np.isin(preds, valid))[0]
            cells_to_agg = [int(cell) for cell in cells_to_agg] # Ensure cells_to_agg is in integers (if not already)

            # Assign the higher-level class label to the identified cells
            query.obs.loc[cells_to_agg, "predicted_" + higher_level_key] = higher_class

    return query

def eval(query, ref_keys, **kwargs):
    class_metrics = defaultdict(lambda: defaultdict(dict))
    for key in ref_keys:
        
        #class_labels = query.obs[key].unique()
        
        threshold = kwargs.get('threshold', True)  # Or some other default value    
        class_labels = query.obs[key].unique()
        pred_classes = query.obs["predicted_" + key].unique()
        labels = list(set(class_labels).union(set(pred_classes)))

        #if threshold:
            ## If you want to remove "unknown" classes from both true and predicted labels
            #valid_indices = (query.obs["predicted_" + key] != "unknown")
    
    ## Filter the data to remove the "unknown" class
            #true_labels = query.obs[key][valid_indices]
            #predicted_labels = query.obs["predicted_" + key][valid_indices]
        #else:
        true_labels= query.obs[key]
        predicted_labels = query.obs["predicted_" + key]
            
    # Calculate accuracy and confusion matrix after removing "unknown" labels
        accuracy = accuracy_score(true_labels, predicted_labels)

        conf_matrix = confusion_matrix(
            true_labels, predicted_labels, 
            labels=labels
        )
        class_metrics[key]["confusion"] = {
            "matrix": conf_matrix,
            "labels": labels
            #"accuracy": accuracy
        }
        # Classification report for predictions
        class_metrics[key]["classification_report"] = classification_report(true_labels, predicted_labels, 
                                        labels=labels,output_dict=True, zero_division=np.nan)
       
    return class_metrics

def update_classification_report(report):
    for label, metrics in report.items():
        if metrics['support'] == 0:
            metrics['recall'] = "nan" 
            metrics['f1-score'] = "nan" 
    return report

def plot_confusion_matrix(query_name, ref_name, key, confusion_data, output_dir):
    new_query_name = query_name.replace(" ", "_").replace("/", "_").replace("(","").replace(")","")
    new_ref_name = ref_name.replace(" ", "_").replace("/", "_").replace("(","").replace(")","")
               
    # Extract confusion matrix and labels from the confusion data
    conf_matrix = confusion_data["matrix"]
    labels = confusion_data["labels"]

    # Plot the confusion matrix
    plt.figure(figsize=(20, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Reds', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {query_name} vs {ref_name} - {key}')
    
    # Save the plot
   # output_dir = os.path.join(projPath, 'results', 'confusion')
    
    os.makedirs(os.path.join(output_dir, new_query_name, new_ref_name), exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(os.path.join(os.path.join(output_dir, new_query_name, new_ref_name),f"{key}_confusion.png"))
    plt.close() 



def plot_roc_curves(metrics, title="ROC Curves for All Keys and Classes", save_path=None):
    """
    Plots ROC curves for each class at each key level from the metrics dictionary on the same figure.
    
    Parameters:
    metrics (dict): A dictionary with structure metrics[key][class_label] = {tpr, fpr, auc, optimal_threshold}.
    title (str): The title of the plot.
    save_path (str, optional): The file path to save the plot. If None, the plot is not saved.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_keys = len(metrics)
    plt.figure(figsize=(10, 8))
    plt.suptitle(title, fontsize=25)

    # Create a subplot for each key
    for idx, key in enumerate(metrics.keys()):
        plt.subplot(num_keys, 1, idx + 1)
        plt.title(f"ROC Curves for {key}")
        
        # Plot ROC curves for each class under the current key
        for class_label in metrics[key]:
            if isinstance(metrics[key][class_label], dict):
                if all(k in metrics[key][class_label] for k in ["tpr", "fpr", "auc"]):
                    tpr = metrics[key][class_label]["tpr"]
                    fpr = metrics[key][class_label]["fpr"]
                    roc_auc = metrics[key][class_label]["auc"]

                    # Find the index of the optimal threshold
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_fpr = fpr[optimal_idx]
                    optimal_tpr = tpr[optimal_idx]

                    # Plot the ROC curve for the current class
                    plt.plot(fpr, tpr, lw=2, label=f"Class {class_label} (AUC = {roc_auc:.3f})")
                    # Plot the optimal threshold as a point
                    plt.scatter(optimal_fpr, optimal_tpr, color='red', marker='o')

        # Plot the reference line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)

    # Adjust the layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot to the given path if provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    



def combine_f1_scores(class_metrics, ref_keys):
   # metrics = class_metrics
    # Dictionary to store DataFrames for each key
    all_f1_scores = {}
    #cutoff = class_metrics["cutoff"]
    # Iterate over each key in ref_keys
    for key in ref_keys:
        # Create a list to store F1 scores for each query-ref combo
        f1_data = [] 
        # Iterate over all query-ref combinations
        for query_name in class_metrics:
            for ref_name in class_metrics[query_name]:
                # Extract the classification report for the current query-ref-key combination
                classification_report = class_metrics[query_name][ref_name][key]["classification_report"]
                # Extract F1 scores for each label
                if classification_report:
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
                                    'weighted_f1': classification_report.get('weighted avg', {}).get('f1-score', None)#,
                                   # 'accuracy': classification_report.get('accuracy', )
                                })

        # Create DataFrame for the current key
        df = pd.DataFrame(f1_data)

        # Store the DataFrame in the dictionary for the current key
        all_f1_scores[key] = df

    return all_f1_scores


def plot_f1_heatmaps(all_f1_scores, threshold, outpath, ref_keys):
    os.makedirs(outpath, exist_ok=True) 
    # Create a figure to hold the plots
    fig, axes = plt.subplots(ncols=len(all_f1_scores), nrows=1, figsize=(40, 10))
    
    for idx, (key, df) in enumerate(all_f1_scores.items()):
        for query in df['query'].unique():
        # Pivot the DataFrame to get references as rows and labels + queries as columns
            pivot_df = df.pivot_table(index='reference', columns='label', values='f1_score')

        # Plot heatmap for label-level F1 scores with flipped axes
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', cbar_kws={'label': 'F1 Score'}, ax=axes[idx])
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90)
            axes[idx].set_title(f'F1 Scores for {key} at threshold = {threshold:.2f}', fontsize=25)
            axes[idx].set_ylabel('Reference', fontsize=30)
            axes[idx].set_xlabel('Label', fontsize=30)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(os.path.join(outpath,'label_f1_scores.png'))  # Change the file name as needed
    plt.close()
     
 ## Now create a final heatmap for macro, micro, and weighted F1 scores
    final_f1_data = pd.DataFrame()
    for key, df in all_f1_scores.items():
        macro = df.drop(columns=['label', 'f1_score'])
        macro["key"] = key
        final_f1_data = pd.concat([final_f1_data, macro], ignore_index=True)
    
    # Step 1: Aggregate data for the weighted F1 score
    weighted_f1_data = final_f1_data[['reference', 'key', 'query', 'weighted_f1']]

    # Step 2: Pivot data to structure it for heatmap plotting
    pivot_f1 = weighted_f1_data.pivot_table(
        index='reference',
        columns=['key', 'query'],
        values='weighted_f1'
    )
    ordered_columns = [col for col in ref_keys if col in pivot_f1.columns.get_level_values('key').unique()]
    pivot_f1 = pivot_f1[ordered_columns]

    # Step 3: Create a new figure for the aggregated F1 score heatmap
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot the heatmap for the weighted F1 score
    sns.heatmap(pivot_f1, annot=True, cmap='YlOrRd', cbar_kws={'label': 'Weighted F1 Score'}, fmt='.3f', ax=ax)
    new_column_labels = [label[0] for label in pivot_f1.columns]
    ax.set_xticklabels(new_column_labels, rotation=45, ha="right")
    # Set the title and labels
    ax.set_title(f'Weighted F1 Score at threshold = {threshold:.2f}', fontsize=30)
    ax.set_xlabel('Key and Query', fontsize=15)
    ax.set_ylabel('Reference', fontsize=15)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability

    plt.tight_layout()
    plt.savefig(os.path.join(outpath,"agg_f1_scores.png"))


        
def get_test_data(census_version, test_name, subsample=10, 
                  organism="homo_sapiens", 
                  split_key="dataset_title"):
    census = cellxgene_census.open_soma(census_version=census_version)
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = cellxgene_census.get_obs(census, organism,
        value_filter=(
            "tissue_general == 'brain' and "
            "is_primary_data == True and "
            "tissue == 'frontal cortex' " # putting this in to speed things up for docker
        ))
    
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id", suffixes=(None,"_y"))
    brain_obs.drop(columns=['soma_joinid_y'], inplace=True)
    # Filter based on organism
    test_obs = brain_obs[brain_obs[split_key].isin([test_name])]
    filtered_ids = test_obs["soma_joinid"]
    subsample_ids = subsample_cells(brain_obs, filtered_ids, subsample=subsample)
    # Adjust organism naming for compatibility
    organism_name_mapping = {
        "homo_sapiens": "Homo sapiens",
        "mus_musculus": "Mus musculus"
    }
    organism = organism_name_mapping.get(organism, organism)
    cell_columns = [
        "assay", "cell_type", "tissue",
        "tissue_general", "suspension_type",
        "disease", "dataset_id", "development_stage",
        "soma_joinid"
    ]
    

    random.seed(1)
    test = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
           # obs_value_filter= "development_stage"  
           # need to filter out fetal potentially?
            var_value_filter = "nnz > 50",
          #  obs_column_names=cell_columns,
            obs_coords=subsample_ids)
    test.obs= test.obs.merge(dataset_info,  on="dataset_id", suffixes=(None,"_y"))
    columns_to_drop = [col for col in test.obs.columns if col.endswith('_y')]
    test.obs.drop(columns=columns_to_drop, inplace=True)
    return test

def split_anndata_by_obs(adata, obs_key="dataset_title"):
    """
    Split an AnnData object into multiple AnnData objects based on unique values in an obs key.

    Parameters:
    - adata: AnnData object to split.
    - obs_key: Key in `adata.obs` on which to split the data.

    Returns:
    - A dictionary where keys are unique values in `obs_key` and values are corresponding AnnData subsets.
    """
    # Dictionary comprehension to create a separate AnnData for each unique value in obs_key
    split_data = {
        value: adata[adata.obs[obs_key] == value].copy() 
        for value in adata.obs[obs_key].unique()
    }
    
    return split_data


