#!/usr/bin/env nextflow


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

process mapQuery {
    input:
    val model_path
    path relabel_q
    path query_file
    val batch_key

    output:
    path "${query_file.toString().replace('.h5ad','_processed.h5ad')}"

script:

"""

python $projectDir/bin/process_query.py \\
                        --model_path ${model_path} \\
                        --relabel_path ${relabel_q} \\
                        --query_path ${query_file} \\
                        --batch_key ${batch_key}
"""

}


// process getRefs {
    // input:
    // val organism
    // val census_version
    // val subsample_ref
    // val relabel_r

    // output:
    // path "refs/*.h5ad", emit: ref_paths


    // script:
    // """
    // # Run the python script to generate the files
    // python $projectDir/bin/get_refs.py --organism ${organism} --census_version ${census_version} --subsample_ref ${subsample_ref} --relabel_path ${relabel_r}

    // # After running the python script, all .h5ad files will be saved in the refs/ directory inside a work directory
    // """
// }

process rfClassify {

    publishDir "${params.outdir}", mode: "copy"

    input:
    val tree_file
    val query_path
    path ref_path
    val ref_keys
    val cutoff

    output:
    path "f1_results/*f1.scores.tsv", emit: f1_score_channel  // Match TSV files in f1_results
    path "roc/*.tsv", emit: auc_channel
    path "roc/**"
    path "confusion/**"
    path "probs/**"
    path "probs/*tsv"
    path "predicted_meta/*tsv"

  //  publish:



    script:
    """
    python $projectDir/bin/rfc_classify.py --tree_file ${tree_file} --query_path ${query_path} --ref_path ${ref_path} --ref_keys ${ref_keys} \\
                --cutoff ${cutoff}
 
    """

}

process plot_auc_dist {
    publishDir "${params.outdir}", mode: "copy"

    input:
    file auc

    output:
    path "dists/*distribution.png" // Wildcard to capture all relevant output files

    script:
    
    """
    python $projectDir/bin/plot_auc_dist.py --roc_paths ${auc}
 
    """ 
}

process plot_f1_results {
    publishDir "${params.outdir}", mode: "copy"

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

    Channel.fromPath(params.queries)
    .set{ query_paths }

    Channel.fromPath(params.refs)
    // .collect() 
    .set { ref_paths }
    
    processed_queries = mapQuery(model_path, params.relabel_q, query_paths, params.batch_key) 

    // Pass each file in ref_paths to rfc_classify using one query file at a time
    rfClassify(params.tree_file, processed_queries.first(), ref_paths, params.ref_keys.join(' '), params.cutoff)


    // Collect all individual output files into a single channel
    auc = rfClassify.out.auc_channel
    f1_scores = rfClassify.out.f1_score_channel

    plot_auc_dist(auc.flatten().toList())

    // Plot f1 score heatmaps using a list of file names from the f1 score channel
    plot_f1_results(params.ref_keys.join(' '), params.cutoff, f1_scores.flatten().toList()) 
}

workflow.onComplete {
    println "Successfully completed"
    /*
    // This bit cannot be run interactively????, only try when sending as pipeline 
    jsonStr = JsonOutput.toJson(params)
    file("${params.outdir}/params.json").text = JsonOutput.prettyPrint(jsonStr)
    */
    println ( workflow.success ? 
    """
    ===============================================================================
    Pipeline execution summary
    -------------------------------------------------------------------------------

    Run as      : ${workflow.commandLine}
    Started at  : ${workflow.start}
    Completed at: ${workflow.complete}
    Duration    : ${workflow.duration}
    Success     : ${workflow.success}
    workDir     : ${workflow.workDir}
    Config files: ${workflow.configFiles}
    exit status : ${workflow.exitStatus}

    --------------------------------------------------------------------------------
    ================================================================================
    """.stripIndent() : """
    Failed: ${workflow.errorReport}
    exit status : ${workflow.exitStatus}
    """.stripIndent()
    )
}

workflow.onError = {
println "Error: something went wrong, check the pipeline log at '.nextflow.log"
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/  