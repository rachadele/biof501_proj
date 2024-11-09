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
import yaml
from pathlib import Path
current_directory = Path.cwd()
projPath = current_directory.parent

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


def extract_data(data, filtered_ids, subsample=500, organism=None, census=None, 
    obs_filter=None, cell_columns=None, dataset_info=None, dims=20, relabel_path=f"{projPath}/meta/relabel/census_map_human.tsv"):
    
    brain_cell_subsampled_ids = subsample_cells(data, filtered_ids, subsample)
    # Assuming get_seurat is defined to return an AnnData object
    adata = cellxgene_census.get_anndata(
        census=census,
        organism=organism,
        obs_value_filter=obs_filter,  # Ensure this is constructed correctly
        obs_column_names=cell_columns,
        obs_coords=brain_cell_subsampled_ids,
        var_value_filter = "nnz > 20",
        obs_embeddings=["scvi"])
    
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
   # newmeta = newmeta.drop(columns=['soma_joinid_y']).rename(columns={'soma_joinid_x': 'soma_joinid'})
    adata.obs = newmeta
    # Assuming relabel_wrapper is defined
    adata = relabel(adata, relabel_path=relabel_path, join_key="cell_type", sep='\t')
    # Convert all columns in adata.obs to factors (categorical type in pandas)
    return adata

def split_and_extract_data(data, split_column, subsample=500, organism=None, census=None, cell_columns=None, dataset_info=None, dims=20):
    # Get unique split values from the specified column
    unique_values = data[split_column].unique()
    refs = {}

    for split_value in unique_values:
        # Filter the brain observations based on the split value
        filtered_ids = data[data[split_column] == split_value]['soma_joinid'].values
        obs_filter = f"{split_column} == '{split_value}'"
        
        adata = extract_data(data, filtered_ids, subsample, organism, census, obs_filter, 
                             cell_columns, dataset_info, dims=dims)
        dataset_titles = adata.obs['dataset_title'].unique()

        if len(dataset_titles) > 1:
            name_to_use = split_value
        else:
            name_to_use = dataset_titles[0]

        refs[name_to_use] = adata

    return refs

def get_census(census_version="2024-07-01", organism="homo_sapiens", subsample=500, split_column="tissue", dims=20, projPath=projPath):

    census = cellxgene_census.open_soma(census_version=census_version)
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = cellxgene_census.get_obs(census, organism,
        value_filter=(
            "tissue_general == 'brain' and "
            "is_primary_data == True and "
            "disease == 'normal'"
        ))
    
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id", suffixes=(None,"_y"))
    brain_obs.drop(columns=['soma_joinid_y'], inplace=True)
    # Filter based on organism
    if organism == "homo_sapiens":
        brain_obs_filtered = brain_obs[
            brain_obs['collection_name'].isin([
                "SEA-AD: Seattle Alzheimer’s Disease Brain Cell Atlas",
                "Transcriptomic cytoarchitecture reveals principles of human neocortex organization"
            ])
        ]
        brain_obs_filtered = brain_obs_filtered[~brain_obs_filtered['cell_type'].isin(["unknown", "glutamatergic neuron"])]
    elif organism == "mus_musculus":
        brain_obs_filtered = brain_obs[
            brain_obs['collection_name'].isin(["A taxonomy of transcriptomic cell types across the isocortex and hippocampal formation"]) 
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
        dataset_info=dataset_info, dims=dims
    )
    # Get embeddings for all data together
    filtered_ids = brain_obs_filtered['soma_joinid'].values
    adata = extract_data(
        brain_obs_filtered, filtered_ids,
        subsample=subsample, organism=organism,
        census=census, obs_filter=None,
        cell_columns=cell_columns, dataset_info=dataset_info, dims=dims
    )
    refs["whole cortex"] = adata

    for name, ref in refs.items():
        dataset_title = name.replace(" ", "_")
        for col in ref.obs.columns:
            if ref.obs[col].dtype.name =='category':
    # Convert to Categorical and remove unused categories
                ref.obs[col] = pd.Categorical(ref.obs[col].cat.remove_unused_categories())
            #ref.obs[col].inplace=True
        #p = sc.pl.umap(ref, color=["rachel_subclass", "tissue","collection_name"])
        #breakpoint
        # sc.savefig(f"{projPath}/refs/census/{dataset_title}_{subsample}_umap.png", dpi=300, bbox_inches='tight')

        meta = ref.obs[["cell_type", "rachel_class", "rachel_subclass", "rachel_family"]].drop_duplicates()
        meta.to_csv(f"{projPath}/meta/relabel/{dataset_title}_relabel.tsv", sep="\t", index=False)

    return refs



def process_query(query, tissue="frontal cortex", dataset_id="QUERY", dataset_title=None, batch_key="sample", projPath=projPath):
    # Ensure the input AnnData object is valid
    if not isinstance(query, ad.AnnData):
        raise ValueError("Input must be an AnnData object.")

    # Assign ensembl_id to var
    #query.var["ensembl_id"] = query.var["feature_id"]
    if "feature_id" in query.var.columns:
        query.var.set_index("feature_id", inplace=True)

    model_file_path=os.path.join(projPath, "scvi-human-2024-07-01")
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


def classify_cells(ref, query, ref_keys):
    #   ref,query = map_valid_labels(ref, query, tree, ref_keys)
    probabilities = {}
    for key in ref_keys:
        probabilities[key]={}
        # Train the random forest classifier on the census data
        rfc = RandomForestClassifier()
        rfc.fit(ref.obsm["scvi"], ref.obs[key].values)
        
        # Predict probabilities for each class in the query data
        probs = rfc.predict_proba(query.obsm["scvi"])
        class_labels = rfc.classes_
        probabilities[key]["probabilities"] = probs
        probabilities[key]["class_labels"]=class_labels
       # probabilities[key]["optimal_thresholds"]
    return(probabilities)

def evaluate_classifier(probabilities, query, ref_keys, specified_threshold=None):
    optimal_thresholds = {}
    metrics={}
    for key in ref_keys:
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
                fpr, tpr, thresholds = roc_curve(true_labels[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
            #   plt.plot(fpr, tpr, lw=2, label=f"Class {class_label} (AUC = {roc_auc:.2f})")
                
                # Optimal threshold based on Youden's J statistic, or use the specified threshold
                if specified_threshold is not None:
                    optimal_threshold = specified_threshold
                else:
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[optimal_idx]
                    if optimal_threshold == float('inf'):
                        optimal_threshold = 0 
                optimal_thresholds[key][class_label]=optimal_threshold
                metrics[key][class_label]["tpr"] = tpr
                metrics[key][class_label]["fpr"] = fpr
                metrics[key][class_label]["auc"] = roc_auc
                metrics[key][class_label]["optimal_threshold"] = optimal_threshold
        
            # Assign predictions based on optimal thresholds
        predicted_classes = []
        
        average_threshold= get_average_threshold(optimal_thresholds, key)
        metrics[key]["average_threshold"] = average_threshold
                
        for i in range(query.n_obs):
            class_probs = probs[i]
            max_class = "unknown"
            max_prob = 0.0
            for j, class_label in enumerate(class_labels): 
                if class_probs[j] >= average_threshold and class_probs[j] > max_prob:
                    max_class = class_label
                    max_prob = class_probs[j]
            #max_probs.append(max_prob)
            predicted_classes.append(max_class)
        
        # Store predictions and confidence in `query`
        query.obs["predicted_" + key] = predicted_classes
        query.obs["confidence"] = np.max(probs, axis=1)
        
        true_labels = query.obs[key].unique()
        predicted_labels = query.obs["predicted_" + key].unique()

        missing_true = [label for label in class_labels if label not in true_labels]
        missing_predicted = [label for label in class_labels if label not in predicted_labels]

        if missing_true:
            print(f"Warning: Missing classes in true labels: {missing_true}. Removing from confusion.")
        if missing_predicted:
            print(f"Missing classes in predicted labels: {missing_predicted}") 
        
        # remove labels that are not in the original author labels
        labels=[label for label in class_labels if label not in missing_true]
    # Classification report for predictions
        metrics[key]["classification_report"] = classification_report(query.obs[key], query.obs["predicted_" + key], 
                                        labels=labels,output_dict=True)
        
        # Plot UMAP with classified cell types
    #sc.pl.umap(query, color=["confidence"] + ["predicted_" + key for key in ref_keys], ncols=1, na_in_legend=True, legend_fontsize=20)

    return query,metrics

def get_average_threshold(optimal_thresholds, key):
    """
    Calculates the average threshold for a given key across all class labels.
    
    Parameters:
    optimal_thresholds (dict): A dictionary containing optimal thresholds for each key and class label.
    key (str): The specific key for which to average the thresholds.
    
    Returns:
    float: The average threshold for the given key.
    """
    thresholds = [optimal_thresholds[key][class_label] for class_label in optimal_thresholds[key] if optimal_thresholds[key][class_label] is not None]
    average_threshold = sum(thresholds) / len(thresholds) if thresholds else 0.5  # Default to 0.5 if no thresholds
    return average_threshold 

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
    plt.figure(figsize=(15, 20))
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
    
    plt.show()

import pandas as pd

def combine_f1_scores(results, ref_keys):
    # Dictionary to store DataFrames for each key
    all_f1_scores = {}
    
    # Iterate over each key in ref_keys
    for key in ref_keys:
        # Create a list to store F1 scores for each query-ref combo
        f1_data = []
        
        # Iterate over all query-ref combinations
        for query_name in results:
            for ref_name in results[query_name]:
                # Extract the classification report for the current query-ref-key combination
                classification_report = results[query_name][ref_name][key]["classification_report"]
                
                # Extract F1 scores for each label
                if classification_report:
                    for label, metrics in classification_report.items():
                        if isinstance(metrics, dict) and 'f1-score' in metrics:
                            f1_data.append({
                                'query': query_name,
                                'reference': ref_name,
                                'label': label,
                                'f1_score': metrics['f1-score'],
                                'macro_f1': classification_report.get('macro avg', {}).get('f1-score', None),
                                'micro_f1': classification_report.get('micro avg', {}).get('f1-score', None),
                                'weighted_f1': classification_report.get('weighted avg', {}).get('f1-score', None)
                            })

        # Create DataFrame for the current key
        df = pd.DataFrame(f1_data)

        # Store the DataFrame in the dictionary for the current key
        all_f1_scores[key] = df

    return all_f1_scores


def plot_f1_heatmaps(all_f1_scores):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Create a figure to hold the plots
    fig, axes = plt.subplots(nrows=1, ncols=len(all_f1_scores)+1, figsize=(20, 10))
    
    # Iterate over the F1 score DataFrames for each key
    for idx, (key, df) in enumerate(all_f1_scores.items()):
        # Pivot the DataFrame to get labels as rows and queries + references as columns
        pivot_df = df.pivot_table(index='label', columns=['query', 'reference'], values='f1_score')
        
        # Plot heatmap for label-level F1 scores
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'F1 Score'}, ax=axes[idx])
        axes[idx].set_title(f'F1 Scores for {key} (Label-level)', fontsize=16)
        axes[idx].set_xlabel('Query - Reference', fontsize=12)
        axes[idx].set_ylabel('Label', fontsize=12)
        
    # Now create a final heatmap for macro, micro, and weighted F1 scores
    final_f1_data = []
    for key, df in all_f1_scores.items():
        final_f1_data.append({
            'key': key,
            'macro_f1': df['macro_f1'].iloc[0] if not df['macro_f1'].isnull().all() else None,
            'micro_f1': df['micro_f1'].iloc[0] if not df['micro_f1'].isnull().all() else None,
            'weighted_f1': df['weighted_f1'].iloc[0] if not df['weighted_f1'].isnull().all() else None
        })
    
    final_f1_df = pd.DataFrame(final_f1_data).set_index('key')
    
    # Plot the final heatmap for macro, micro, and weighted F1 scores
    sns.heatmap(final_f1_df.T, annot=True, cmap='coolwarm', cbar_kws={'label': 'F1 Score'}, ax=axes[-1])
    axes[-1].set_title('Macro, Micro, and Weighted F1 Scores', fontsize=16)
    axes[-1].set_xlabel('Key', fontsize=12)
    axes[-1].set_ylabel('F1 Score Type', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

        
def get_test_data(census_version, test_name, subsample=500, 
                  organism="homo_sapiens", 
                  split_key="dataset_title"):
    census = cellxgene_census.open_soma(census_version=census_version)
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = cellxgene_census.get_obs(census, organism,
        value_filter=(
            "tissue_general == 'brain' and "
            "is_primary_data == True"
        ))
    
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id", suffixes=(None,"_y"))
    brain_obs.drop(columns=['soma_joinid_y'], inplace=True)
    # Filter based on organism
    test_obs = brain_obs[brain_obs[split_key].isin([test_name])]
    subsample_ids = random.sample(list(test_obs["soma_joinid"]), subsample)
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
            var_value_filter = "nnz > 20",
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
