# Patch Clamp Data Analysis

This repository provides a set of tools implemented in Python for the analysis of patch clamp data, as described in the paper *[Insert Paper Title]*. The primary component is the `functions.py` module, which contains a `patchclampdata` class object designed to facilitate thresholding and plotting of patch clamp data from CSV files. While the current implementation is a first draft, it aims to serve as a foundational framework for researchers interested in conducting similar analyses.

## Contents

- **functions.py**: This module houses the `patchclampdata` object, designed to take a CSV file containing patch clamp data and provide functionality for thresholding and plotting. While currently optimized for fixed voltage positive current experiments, it can be adapted for broader use cases. Note that there is room for code cleaning and optimization.

- **demo.ipynb**: A Jupyter Notebook demonstrating the usage of the functions provided in `functions.py`. This notebook showcases how the functions were utilized to generate figures for the associated paper.

- **data.csv**: An example CSV file containing patch clamp data. This dataset is for use in `demo.ipynb`.

- **organise_patchclamp_files.ipynb**: This notebook outlines the process of organizing .asc files into separate folders for each experiment ID.

- **asc_to_csv_conversion.ipynb**: Details the conversion process from ASC files to CSV format. 

## Comments

This is very much a proof of concept and though it can generate the figures there is room for improvment in code quality, commenting and implimnetation.

We hope this will be useful as a start point for people interseted in analysising patch clamp data in python.

