import pandas as pd
import numpy as np
import scanpy as sc
import random
import cellxgene_census
import cellxgene_census[experimental]
projPath = "."
import os
import anndata as ad
import scvi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


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

def relabel(anndata_obj, relabel_path):
    # Read the relabel table from the file
    relabel_df = pd.read_csv(relabel_path, sep="\t")  # Adjust the separator as needed
    # Take the first column as the join key
    join_key = relabel_df.columns[0]
    # Ensure the join_key is in both the AnnData object and the relabel DataFrame
    if join_key not in anndata_obj.obs.columns:
        raise ValueError(f"{join_key} not found in AnnData object observations.")
    if join_key not in relabel_df.columns:
        raise ValueError(f"{join_key} not found in relabel DataFrame.")
    # Perform the left join to update the metadata
    meta = anndata_obj.obs.merge(relabel_df, on=join_key, how='left')
    # Update the AnnData object's obs with the new metadata
    anndata_obj.obs = meta.set_index(anndata_obj.obs.index)
    return anndata_obj


def extract_data(data, filtered_ids, subsample=500, organism=None, census=None, 
                 obs_filter=None, cell_columns=None, dataset_info=None, dims=20, relabel_path=f"{projPath}/meta/relabel/census_map_human.tsv"):
    
    brain_cell_subsampled_ids = subsample_cells(data, filtered_ids, subsample)
    # Assuming get_seurat is defined to return an AnnData object
    adata = get_anndata(
        census=census,
        organism=organism,
        obs_value_filter=obs_filter,  # Ensure this is constructed correctly
        obs_column_names=cell_columns,
        obs_coords=brain_cell_subsampled_ids,
        var_index="feature_id",
        obs_embeddings=["scvi"]
        
    )
    print("Subsampling successful.")
    # Filter out genes that are not expressed in at least 3 cells
    adata = adata[adata.X.sum(axis=0) >= 3, :]
    # Preprocessing
   # sc.pp.normalize_total(adata)
    #sc.pp.scale(adata)
   # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
  #  sc.tl.pca(adata, n_comps=dims)
  #  sc.tl.umap(adata)
    # Merging metadata with dataset_info
    newmeta = adata.obs.merge(dataset_info, on="dataset_id", suffixes=(None,"y"))
   # newmeta = newmeta.drop(columns=['soma_joinid_y']).rename(columns={'soma_joinid_x': 'soma_joinid'})
    adata.obs = newmeta
    # Assuming relabel_wrapper is defined
    adata = relabel(adata, relabel_path=relabel_path)
    return adata

def split_and_extract_data(data, split_column, subsample=500, organism=None, census=None, cell_columns=None, dataset_info=None, dims=20):
    # Get unique split values from the specified column
    unique_values = data[split_column].unique()
    refs = {}

    for split_value in unique_values:
        # Filter the brain observations based on the split value
        filtered_ids = data[data[split_column] == split_value]['soma_joinid'].values
        obs_filter = f"{split_column} == '{split_value}'"
        
        adata = extract_data(data, filtered_ids, subsample, organism, census, obs_filter, cell_columns, dataset_info, dims=dims)
        dataset_titles = adata.obs['dataset_title'].unique()

        if len(dataset_titles) > 1:
            name_to_use = split_value
        else:
            name_to_use = dataset_titles[0]

        refs[name_to_use] = adata

    return refs

def get_census(organism="homo_sapiens", census_version="2024-07-01", subsample=500, split_column="dataset_id", dims=20):
    census = open_soma(census_version)
    
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = census.get("census_data").get("homo_sapiens").get("obs").read(
        value_filter="tissue_general == 'brain' & is_primary_data == True & disease == 'normal'",
        column_names=[
            "assay", "cell_type", "sex", "tissue", "tissue_general", "suspension_type",
            "disease", "dataset_id", "development_stage", "soma_joinid"
        ]
    ).concat().to_pandas()
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id")
    brain_obs.rename(columns={'soma_joinid_x': 'soma_joinid'}, inplace=True)
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

    # Set random seed for reproducibility
    random.seed(1)

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
    
    refs = split_and_extract_data(
        brain_obs_filtered, split_column=split_column,
        subsample=subsample, organism=organism,
        census=census, cell_columns=cell_columns,
        dataset_info=dataset_info, dims=dims
    )

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
        p = sc.pl.umap(ref, color=["rachel_subclass", "assay", "tissue", "dataset_title"], show=False)
        sc.savefig(f"{projPath}/refs/census/{dataset_title}_{subsample}_umap.png", dpi=300, bbox_inches='tight')

        meta = ref.obs[["cell_type", "rachel_class", "rachel_subclass", "rachel_family"]].drop_duplicates()
        meta.to_csv(f"{projPath}/meta/relabel/{dataset_title}_relabel.tsv", sep="\t", index=False)

    return refs



def process_adat_query(adata_query, proj_path, tissue="prefrontal cortex", dataset_id="QUERY", dataset_title=None, batch_key="sample",
                        model_file_path=os.path.join(projPath, "scvi-human-2024-07-01")):
    # Ensure the input AnnData object is valid
    if not isinstance(adata_query, ad.AnnData):
        raise ValueError("Input must be an AnnData object.")

    # Assign ensembl_id to var
    adata_query.var["ensembl_id"] = adata_query.var.index
    adata_query.obs["n_counts"] = adata_query.X.sum(axis=1)
    adata_query.obs["joinid"] = list(range(adata_query.n_obs))
    adata_query.obs["batch"] = adata_query.obs[batch_key]

    # Filter out missing HGNC features
    adata_query = adata_query[:, adata_query.var["gene_name"].notnull().values].copy()

    # Prepare the query AnnData for scVI
    scvi.model.SCVI.prepare_query_anndata(adata_query, model_file_path)
    vae_q = scvi.model.SCVI.load_query_data(adata_query, model_file_path)

    # Set the model to trained and get latent representation
    vae_q.is_trained = True
    latent = vae_q.get_latent_representation()
    adata_query.obsm["scvi"] = latent

    # Add tissue information and dataset identifiers
    adata_query.obs["tissue"] = tissue
    adata_query.obs["tissue_general"] = "brain"
    adata_query.var_names = adata_query.var["gene_id"]
    adata_query.obs["dataset_id"] = dataset_id
    adata_query.obs["dataset_title"] = dataset_title

    # Normalize and log-transform the data
    sc.pp.normalize_total(adata_query, target_sum=1e4)
    sc.pp.log1p(adata_query)

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata_query, n_neighbors=30, use_rep="scvi")
    sc.tl.umap(adata_query)
    sc.tl.leiden(adata_query)

    return adata_query

# Example usage
# lau_sub = ad.read_h5ad("/space/grp/rschwartz/rschwartz/census-stuff/h5ad/lau_updated_subsampled_1000.h5ad")
# processed_data = process_adat_query(lau_sub, proj_path="/path/to/project")

def classify_cells(adata_census, adata_query, ref_keys): 
    for key in ref_keys:
        rfc = RandomForestClassifier()
        rfc.fit(adata_census.obsm["scvi"], adata_census.obs[key].values)
        adata_query.obs["transfer_" + key] = rfc.predict(adata_query.obsm["scvi"])

        # let's get confidence scores
        probabilities = rfc.predict_proba(adata_query.obsm["scvi"])

        confidence = np.zeros(adata_query.n_obs)
        for i in range(adata_query.n_obs):
            confidence[i] = probabilities[i][rfc.classes_ == adata_query.obs["predicted_cell_type"][i]]
            
        confidence = np.max(probabilities, axis=1)

        # Add confidence to adata and threshold prediction
        adata_query.obs["confidence"] = confidence
        
        #adata_query.obs["predicted_cell_type"] = np.where(confidence >= 0.75, adata.obs["predicted_cell_type"], "unknown")
        #sc.pl.umap(adata_query, color=["dataset_title", "predicted_cell_type", "Cell_type"], ncols=1, na_in_legend=True, legend_fontsize=20)
        
        
