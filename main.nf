#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process RunCensus {
    // Define the Docker image to use
    container 'census_pipeline' // replace with your actual Docker image name

    input:
    path input_data

    output:
    path 'output/*' // Adjust as necessary for your output files

    script:
    """
    python census.py --input ${input_data}
    """
}

workflow {
    // Define your workflow, including how to pass input data
    input_data = file('path/to/your/input_data') // Specify your input data here

    RunCensus(input_data)
}
