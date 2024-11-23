#!/user/bin/python3

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import warnings
import adata_functions
from adata_functions import *
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from types import SimpleNamespace

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify cells given 1 ref and 1 query")
    parser.add_argument('--roc_paths', type=str, nargs="+")

    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def main():
    # Parse command line arguments
    args = parse_arguments()
    roc_paths = args.roc_paths
    roc_df = pd.DataFrame()

# Loop over the list of file paths (roc_paths)
    for file in roc_paths:
        temp_df = pd.read_csv(os.path.join(file),sep="\t")
        roc_df = pd.concat([temp_df, roc_df])

    plot_distribution(roc_df,var="optimal threshold",outdir="dists")
    plot_distribution(roc_df, var="auc",outdir="dists")

if __name__ == "__main__":
    main()