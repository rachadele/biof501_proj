# Evaluation of Multiple Reference Datasets for Cell Type Prediction

## Steps to run



## Background

Single-cell expression is a powerful tool for investigating cell-type-specific differences in gene expression within the context of disease, which can deepen our understanding and suggest avenues for treatment. Large amounts of these data have been collected, offering exciting opportunities for meta-analysis. However, in practice this is challenging in part due to a lack of cell-type annotations in public data repositories such as the Gene Expression Omnibus (GEO), as well as inconsistency in any available annotations [1]. Additionally, any new single-cell data that is generated may not agree with annotations in prior literature due to variation on manual annotation strategies of individual research groups. Single-cell researchers often align their data to "reference datasets" using machine learning classifiers such as KNN, logistic regression, SVM and Random Forest. Most of these classifiers perform well when the reference and query are closely aligned– but which reference of many public datasets to is difficult to predict [2][3]. Additionally, modle and reference performance is impossible to evaluate without first aligning the "ground truth" reference and query labels manuallually.

## Workflow description

This pipeline evalutates a random forest classification task on a toy query dataset given 8 reference datasets with a 3-level cell type hierarchy. The test or "query" data comes from a study of human adult prefrontal cortex [cite], while the references comprise 8 datasets from a popular "atlas" of multiple cortical areas [cite]. The references have been pre-downloaded from the CellxGene Discover Census [cite], and have pre-generated embeddings from a variational autoencoder model [cite] trained on all cells in the CellxGene data corpus. The query data is passed through the pre-trained model, and probabilities for the most granular cell type level are predicted by an RF classifier fitted to the reference embeddings. 
Cell types are predicted given an optional filtering threshold for "unknown" types, and then aggregated using the cell type hierarchy tree into broader labels. This ensures that granular predictions correspond to their higher-level predictions, which may not be the case if we fit a classifier separately at each level. ROC curves for each individual label are plotted, and F1 scores given an optional classification threshold are computed. The pipeline also plots confusion matrices for each label at each level.

### Source code 



### Test data
Toy datasets have been profided in the `reference` and `queries` directories. These data are downsampled to comply with Github and Docker's memory requirements. As such, the evaluation may not be an accurate assessment of classification performance. The threshold has been set to `0` by default.
Importantly, during the pipeline run, query and reference data are mapped to a shared "ground truth" set of hierarchical labels defined in `meta.master_hierarchy.json`. I have generated the mapping files (`census_map_human.tsv` and `gittings_relabel.tsv`) for the purposes of this demo, but a user-supplied query would need to perform this mappin manually. These harmonized labels are used for classification and evaluation.

## DAG

## Sample results

## Repo Structure

## Container

I have built a custom Docker container for use with this pipeline; its configuration can be found in `bin/Dockerfile` and `bin/requirements.txt`. The project directory is mounted to the base directory of the container via the config:

```
 docker { 
  	enabled = true
        runOptions = "-v $projectDir:/biof501_proj$projectDir -m 8g --memory-swap -1"
        temp = 'auto'
   }	 

```
Reading `hdf5` formatted files can be memory intensive; I suggest keeping the memory limit and swap limit as is. 

## References

1. Puntambekar S, Hesselberth JR, Riemondy KA, Fu R. Cell-level metadata are indispensable for documenting single-cell sequencing datasets. Koo BK, editor. PLoS Biol. 2021 May 4;19(5):e3001077.
2. Pasquini G, Rojo Arias JE, Schäfer P, Busskamp V. Automated methods for cell type annotation on scRNA-seq data. Computational and Structural Biotechnology Journal. 2021;19:961–9.
3. Lotfollahi, Mohammad, Yuhan Hao, Fabian J. Theis, and Rahul Satija. “The Future of Rapid and Automated Single-Cell Data Analysis Using Reference Mapping.” Cell 187, no. 10 (May 2024): 2343–58. https://doi.org/10.1016/j.cell.2024.03.009.
