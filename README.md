
# Cross-sectional multimodal evidence (Figure 3)

In Multimodal folder:

- The R script "CogRes_Data.R" predicts cognition using CamCAN data on gray-matter, white-matter and fMRI connectivity, using data in "cog_res_data.csv".

# SSM simulations (Figure 4)

In SSM folder:

- The Matlab script "ssm_fit.m" fits a 2-state SSM to GM thickness and fluid intelligence data in g_gm.dat

- fx_model2.m and gx_model2.m specify the state and observation equations respectively
 
- plot_parameters.m plots priors and posteriors (latter with 95% CI)

- Other SPM12 functions needed are in the toolbox directory (see README.md there)

# HMM simulations (Figure 5)

In HMM folder:

- batch.py calls functions below for a small number of simulations+fits, eg to run within a Python interactive session

- batch_parallel.py calls functions below across multiple cores, eg for big simulations. To run from terminal, you will need a python environment (eg "myenv") with numpy, scipy, time, os and sys libraries. Activate environment with eg "conda activate myenv", then "python3 batch_parallel.py"

- hmm_simulate_data.py generates data with a certain proportion of participants having a certain number of "bursts" of decline (producing Figure 5A in grant proposal)

- hmm_model_comparison.py fits HMMs with range of states (K), calling hmm_fit.py below (producing Figure 5B in grant proposal)

- hmm_fit.py fits an HMM based on linear fits to one variable

- plot_results.py produces Figure 5D in grant proposal