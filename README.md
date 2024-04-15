# ssa_patch_clamp_data_analysis

A repository to share the code used to analysis the patch clamp data in the paper -

functions.py contains:

patchclampdata object - the aim of this object is to take a location of a CSV file contain patch clamp data and add functionallity for thresholding and plotting
    this is very much a first run at somthing like this so there is lots of code cleaning and optimisation that can be made 
    though we hope it can act as a starting point for people implimenting this kind of method of analysis to patch clampe data.
    
The function was written to deal with fixed voltage postive current experiments though has been used to analyse all data in the paper using some work arounds

demo.ipynb:

a jupyter notebook used to show an example of the function is used and the figures for the paper were generated

data.csv:

contains an example of some patch clamp data to use in demo.ipynb

organise_patchclamp_files.ipynb:

how the asc files were moved in to there own folder for each experiment_id (this is specific to this expeimental implimnetation though the method will be transferable)

asc_to_csv_conversion.ipynb:

how the asc file were converted to csv files




This is very much a proof of concept and though it can generate the figures there is room for improvment in code quality, commenting and implimnetation.

We hope this will be useful as a start point for people interseted in analysising patch calmp data in python.
