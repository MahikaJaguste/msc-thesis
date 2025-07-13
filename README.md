# msc-thesis

This repository holds code for the "SPARK – Synergising Personalised AI and RAG-Enhanced Knowledge for recommending treatment strategies in chronic lymphocytic leukemia" research study.

### Code Structure

1. `/bias/`: Add origin tags to dataset, data cleaning, statistical tests to identify difference in characteristic among groups, and visualisations.

2. `/patient_community_project`: 

The `scripts/` folder contains pipeline to:
- Build patient treatment response similarity graph
- Run different community detection algorithms
- Run the hybrid approach (SLPA followed by Leiden refinement).

The `analysis/` folder analyses the communities formed and evaluation of metrics.


3. `/data-collection`: Contains scripts/API calls to load the cBioPortal and clinicaltrials.gov datasets.

4. `/data-exploration`: Notebooks for preliminary data exploration of the downloaded datasets.

5. `/triplet-community-detection`, `/cdlib-trial` and `/community/`: These are deprecated. They were previous attempts to run community detection algorithms on a heterogenous graph (containing all entities as nodes, not just patient nodes) and analyse results.