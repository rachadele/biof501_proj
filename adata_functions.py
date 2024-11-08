import pandas as pd
import numpy as np
import scanpy as sc
import random
import cellxgene_census
import cellxgene_census.experimental
projPath = "."
import os
import anndata as ad
import scvi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import yaml


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
    #sc.tl.leiden(adata_query)
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

def get_census(census_version="2024-07-01", organism="homo_sapiens", subsample=500, split_column="tissue", dims=20):

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
    # Convert to Categorical and remove unused categories
            ref.obs[col] = pd.Categorical(ref.obs[col].cat.remove_unused_categories())
            #ref.obs[col].inplace=True
        p = sc.pl.umap(ref, color=["rachel_subclass", "tissue","collection_name"])
        breakpoint
        # sc.savefig(f"{projPath}/refs/census/{dataset_title}_{subsample}_umap.png", dpi=300, bbox_inches='tight')

        meta = ref.obs[["cell_type", "rachel_class", "rachel_subclass", "rachel_family"]].drop_duplicates()
        meta.to_csv(f"{projPath}/meta/relabel/{dataset_title}_relabel.tsv", sep="\t", index=False)

    return refs



def process_adata_query(adata_query, tissue="frontal cortex", dataset_id="QUERY", dataset_title=None, batch_key="sample",
                        model_file_path=os.path.join(projPath, "scvi-human-2024-07-01")):
    # Ensure the input AnnData object is valid
    if not isinstance(adata_query, ad.AnnData):
        raise ValueError("Input must be an AnnData object.")

    # Assign ensembl_id to var
    #adata_query.var["ensembl_id"] = adata_query.var["feature_id"]
    adata_query.var.set_index("feature_id", inplace=True)
    #adata_query.obs["n_counts"] = adata_query.X.sum(axis=1)
    #adata_query.obs["joinid"] = list(range(adata_query.n_obs))
    #adata_query.obs["batch"] = adata_query.obs[batch_key]

    # Filter out missing HGNC features
    adata_query = adata_query[:, adata_query.var["feature_name"].notnull().values].copy()

    # Prepare the query AnnData for scVI
    scvi.model.SCVI.prepare_query_anndata(adata_query, model_file_path)
    vae_q = scvi.model.SCVI.load_query_data(adata_query, model_file_path)

    # Set the model to trained and get latent representation
    vae_q.is_trained = True
    latent = vae_q.get_latent_representation()
    adata_query.obsm["scvi"] = latent

    # Add tissue information and dataset identifiers
    #adata_query.obs["tissue"] = tissue
    #adata_query.obs["tissue_general"] = "brain"
    #adata_query.var_names = adata_query.var["gene_id"]
    #adata_query.obs["dataset_id"] = dataset_id
    #adata_query.obs["dataset_title"] = dataset_title

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata_query, n_neighbors=30, use_rep="scvi")
    sc.tl.umap(adata_query)
    sc.tl.leiden(adata_query)

    return adata_query



# Function to find the parent label with positive samples
def find_parent_with_positive_samples(label, true_labels, class_labels, tree):
    current_label = label
    while current_label is not None:
        # Check if the current label has any positive samples in the true labels
        class_index = np.where(class_labels == current_label)[0]
        if class_index.size > 0 and true_labels[:, class_index[0]].sum() > 0:
            return current_label  # Found a valid label with positive samples
        
        # Move to the parent label
        current_label = tree.get(current_label)
    
    return None  # No valid parent found


def classify_cells(adata_census, adata_query, ref_keys, specified_threshold=None):
    for key in ref_keys:
        # Train the random forest classifier on the census data
        rfc = RandomForestClassifier()
        rfc.fit(adata_census.obsm["scvi"], adata_census.obs[key].values)
        
        # Predict probabilities for each class in the query data
        probabilities = rfc.predict_proba(adata_query.obsm["scvi"])
        class_labels = rfc.classes_
        
        # Binarize the class labels for multiclass ROC computation
        true_labels = label_binarize(adata_query.obs[key].values, classes=class_labels)
        
        # Find the optimal threshold for each class
        optimal_thresholds = {}
        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(class_labels):
            fpr, tpr, thresholds = roc_curve(true_labels[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {class_label} (AUC = {roc_auc:.2f})")
            
            # Optimal threshold based on Youden's J statistic, or use the specified threshold
            if specified_threshold is not None:
                optimal_threshold = specified_threshold
            else:
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                if optimal_threshold == float('inf'):
                    optimal_threshold = 0 
            optimal_thresholds[class_label] = optimal_threshold

        # Plot ROC for all classes
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) for Each Class")
        plt.legend(loc="lower right")
        plt.show()
        
        # Assign predictions based on optimal thresholds
        predicted_classes = []
        max_probs = []
        for i in range(adata_query.n_obs):
            class_probs = probabilities[i]
            max_class = "unknown"
            max_prob = 0.0
            for j, class_label in enumerate(class_labels):
                if class_probs[j] >= optimal_thresholds[class_label] and class_probs[j] > max_prob:
                    max_class = class_label
                    max_prob = class_probs[j]
            max_probs.append(max_prob)
            predicted_classes.append(max_class)
        
        # Store predictions and confidence in `adata_query`
        adata_query.obs["predicted_" + key] = predicted_classes
        adata_query.obs["confidence"] = np.max(probabilities, axis=1)
        
        # Classification report for predictions
        print(f"\nClassification Report for '{key}' predictions:")
        print(classification_report(adata_query.obs[key], adata_query.obs["predicted_" + key], 
                                    labels=class_labels))

        # Plot UMAP with classified cell types
        sc.pl.umap(adata_query, color=["confidence", "predicted_" + key, key], 
                   ncols=1, na_in_legend=True, legend_fontsize=20)




def find_valid_label(tree, current_label):
    if current_label in tree:
        # If the current label exists, return it
        return tree[current_label]
    
    # Check for parent if the current label is not found at this level
    for key, value in tree.items():
        if isinstance(value, dict):
            # Recurse deeper if there's a nested structure
            result = find_valid_label(value, current_label)
            if result:
                return result
    return None  # If no valid label is found


        
def get_test_data(census_version, test_names, subsample=500):
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
    test_obs = brain_obs[brain_obs['collection_name'].isin(test_names)]
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
    
    # Example usage
    random.seed(1)
    test = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
           # obs_value_filter= "development_stage"  
           # need to filter out fetal potentially?
            var_value_filter = "nnz > 20",
          #  obs_column_names=cell_columns,
            obs_coords=subsample_ids)
   # test= relabel(test, relabel_test_path)
    test.obs= test.obs.merge(dataset_info,  on="dataset_id", suffixes=(None,"_y"))
    columns_to_drop = [col for col in test.obs.columns if col.endswith('_y')]
    test.obs.drop(columns=columns_to_drop, inplace=True)
    test= relabel(test,relabel_path=os.path.join(projPath,"meta","relabel","gittings_relabel.tsv.gz"),
                            join_key="observation_joinid",sep="\t")
    # Remove rows where 'rachel_family' is NA in the 'test' AnnData object
    test = test[~test.obs['rachel_family'].isna(), :]
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

# Define the hierarchical structure as a dictionary with colnames and labels
tree = {
    "GABAergic": {
        "colname": "rachel_family",
        "CGE": {
            "colname": "rachel_class",
            "LAMP5": {
                "colname": "rachel_subclass"
            },
            "VIP": {
                "colname": "rachel_subclass"
            },
            "SNCG": {
                "colname": "rachel_subclass"
            }
        },
        "MGE": {
            "colname": "rachel_class",
            "PVALB": {
                "colname": "rachel_subclass"
            },
            "SST": {
                "colname": "rachel_subclass"
            },
            "Chandelier": {
                "colname": "rachel_subclass"
            }
        }
    },
    "Glutamatergic": {
        "colname": "rachel_family",
        "L2/3-6 IT": {
            "colname": "rachel_class"
        },
        "deep layer non-IT": {
            "colname": "rachel_class",
            "L5 ET": {
                "colname": "rachel_subclass"
            },
            "L5/6 NP": {
                "colname": "rachel_subclass"
            },
            "L6 CT": {
                "colname": "rachel_subclass"
            },
            "L6b": {
                "colname": "rachel_subclass"
            }
        }
    },
    "Non-neuron": {
        "colname": "rachel_family",
        "Oligodendrocyte lineage": {
            "colname": "rachel_class",
            "Oligodendrocyte": {
                "colname": "rachel_subclass"
            },
            "OPC": {
                "colname": "rachel_subclass"
            }
        },
        "Astrocyte": {
            "colname": "rachel_class"
        },
        "Immune/Vasculature": {
            "colname": "rachel_class",
            "Pericyte": {
                "colname": "rachel_subclass"
            },
            "VLMC": {
                "colname": "rachel_subclass"
            },
            "Endothelial": {
                "colname": "rachel_subclass"
            }
        }
    }
}

# Define a recursive function to find a valid label

