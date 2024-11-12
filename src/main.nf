#!/usr/bin/env nextflow

// Define the process to set up the environment and get the model file path
process Setup {
    input:
    val organism from params.organism
    val version from params.version
    file "adata_functions.py"

    output:
    val model_file_path into model_file_ch

    environment 'PYTHONPATH', "${params.adata_functions_path}"

    script:
    """
    python -c '
import sys
sys.path.append("${params.adata_functions_path}")
from adata_functions import setup

# The setup function returns the model file path as a string
model_file = setup("${organism}", "${version}")
print(model_file)  # This will output the model file path as a string
    '
    """
}

// Workflow definition
workflow {
    // Run Setup process to retrieve the model file path
    model_file_ch = Setup()

    // You can use model_file_ch here in subsequent processes
    model_file_ch.subscribe { model_file ->
        println("Received model file path: ${model_file}")
        // Further processing with model_file path
    }
}
