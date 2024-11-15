#!/usr/bin/env nextflow

// Define the required input parameters
params.organism = "homo_sapiens"
params.census_version = "2024-07-01" // version of cellxgene census scvi model and data corpus for reference data
params.tree_file = "$projectDir/meta/master_hierarchy.json" // hierarchy to aggregate predicted classes
params.ref_keys = ["rachel_subclass", "rachel_class", "rachel_family"]  // transferred labels to evaluate
params.test_name = "Frontal cortex samples from C9-ALS, C9-ALS/FTD and age matched control brains" // name of query
params.subsample_ref=10 // number of cells per cell type in ref to sample
params.subsample_query=10 // number of total cells in query to sample
params.results = "$projectDir/results"  // Directory where outputs will be saved
params.relabel_q = "$projectDir/meta/gittings_relabel.tsv.gz" // harmonized label mapping for query
params.relabel_r = "$projectDir/meta/census_map_human.tsv" // harmonized label mapping for references
params.cutoff = 0 // do not threshold class probabilities 

process runSetup {
    input:
    val organism
    val census_version

    output:
    path "scvi-human-${census_version}/"

    script:
    """
    python $projectDir/bin/setup.py --organism ${organism} --census_version ${census_version}
    """
}


process getQuery {
    input:
    val organism
    val census_version
    val model_path
    val subsample_query
    val test_name
    val relabel_q

    output:
    path "${test_name.replace(' ', '_').replace('/', '_')}.h5ad"

script:

"""

python $projectDir/bin/get_query.py --organism ${organism} --census_version ${census_version} \\
                        --model_path ${model_path} \\
                        --subsample_query ${subsample_query} \\
                        --test_name '${test_name}' \\
                        --relabel_path ${relabel_q}
"""

}

process getRefs {
    input:
    val organism
    val census_version
    val subsample_ref
    val relabel_r

    output:
    path "refs/*.h5ad", emit: ref_paths


    script:
    """
    # Run the python script to generate the files
    python $projectDir/bin/get_refs.py --organism ${organism} --census_version ${census_version} --subsample_ref ${subsample_ref} --relabel_path ${relabel_r}

    # After running the python script, all .h5ad files will be saved in the refs/ directory inside a work directory
    """
}

process rfc_classify {

    publishDir "${params.results}", mode: "copy"

    input:
    val organism
    val census_version
    val tree_file
    val query_path
    val ref_path
    val ref_keys
    val cutoff

    output:
    path "f1_results/*f1_scores.tsv", emit: f1_score_channel  // Match TSV files in f1_results
    path "roc/**"
    path "confusion/**"
    script:
    """
    python $projectDir/bin/rfc_classify.py --organism ${organism} --census_version ${census_version} \\
                --tree_file ${tree_file} --query_path ${query_path} --ref_path ${ref_path} --ref_keys ${ref_keys} \\
                --cutoff ${cutoff}
 
    """

}

process plot_results {
    publishDir "${params.results}", mode: "copy"

    input:
    val ref_keys
    val cutoff
    file f1_scores

    output:
    path "f1_plots/*png" // Wildcard to capture all relevant output files

    script:
    
    """
    python $projectDir/bin/plot_f1_results.py --ref_keys ${ref_keys} --cutoff ${cutoff} --f1_results ${f1_scores}
 
    """ 
}

// Workflow definition
workflow {

    // Call the setup process to download the model
    model_path = runSetup(params.organism, params.census_version)
    query_path=getQuery(params.organism, params.census_version, model_path, params.subsample_query, params.test_name, params.relabel_q)
    // You can chain additional processes here as needed
    ref_paths = getRefs(params.organism, params.census_version, params.subsample_ref, params.relabel_r)
        
    
    // Pass each file in ref_paths to rfc_classify
    rfc_classify(params.organism, params.census_version, params.tree_file, query_path, ref_paths.flatMap(), params.ref_keys.join(' '), params.cutoff)

    // Collect all individual output files into a single channel
    f1_scores = rfc_classify.out.f1_score_channel

    // Plot f1 score heatmaps using a list of file names from the f1 score channel
    plot_results(params.ref_keys.join(' '), params.cutoff, f1_scores.flatten().toList()) 
}

// workflow.onComplete {
 //   log.info ( workflow.success ? "\nDone! See results directory at $projectDir/results" )
// }
