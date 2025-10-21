# UCI Bank Marketing

## Data Preprocessing Workflow
- Run data_exploration.R to produce the exploratory plots in images folder
- Run data_preprocessing.R to produce the {train/val/test}_prepared.csv files in data folder.
- Run data_sampling.R to produce the {train/val/test}_final.csv files in data folder.

 The *_prepared.csv files are cleaned, modelling-ready splits without feature selection or class imbalance handling.
 The *_final.csv files are the finalized splits used for production.
