# Data

This project uses the public **Telco Customer Churn** dataset published on Kaggle by BlastChar (IBM sample data).

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (about 977 KB)
- Size: 7,043 customers, 21 variables
- Target: `Churn` (Yes/No)

The raw file is not committed to this repository. To run the analysis:

1. Download the dataset from the Kaggle link above (a free Kaggle account is required).
2. Unzip it and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this `data/` folder.

Running `churn_model.R` also writes the processed dataset to `data/telco_processed.csv`. That file is ignored by git.
