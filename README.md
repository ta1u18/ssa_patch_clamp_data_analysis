# Patch Clamp Data Analysis

This repository contains the code used to run the analysis of patch clamp data, in the paper *Amino acid appended supramolecular self-associating amphiphiles demonstrate dual activity against both MRSA and ovarian cancer*. `functions.py` contains a `patchclampdata` class which thresholds, plots produces summary files of patch clamp data processed using the workflow. This includes automatic finding and plotting of events, summary plots used within the ESI of the paper and interactive plots.

## Contents

- **functions.py**: Contains `patchclampdata` class, which CSV file containing patch clamp data and provides functionality for thresholding and plotting. While currently optimized for fixed voltage positive current experiments, it can be adapted for broader use cases. Note that there is room for code cleaning and optimization.

- **demo.ipynb**: A Jupyter Notebook demonstrating the usage of `functions.py`. This notebook gives a demo of how to generate the figures for the associated paper.

- **data.csv**: An example CSV file containing patch clamp data  for use in `demo.ipynb`.

- **organise_patchclamp_files.ipynb**: Shows how the .asc files were into separate folders in the stucture molecule_id/exp_condition/experiment_id for ananlysis using the `patchclampdata` class.

- **asc_to_csv_conversion.ipynb**: Show the conversion process from ASC files to CSV format. 

## Comments

This is very much a proof of concept and though it can generate the figures there is room for improvment in code quality, commenting and implimentation.

We hope this will be useful as a start point for people interested in analysising patch clamp data in python.

