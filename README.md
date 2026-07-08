# customer-churn-prediction-mlops

A `requirements.txt` file has been added with pinned project dependencies.

## Dataset

This project uses the **Telco Customer Churn** dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download).

### About the dataset

**Context:** *"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs."* (IBM Sample Data Sets)

Each row represents a customer; each column contains customer attributes described in the column metadata. The dataset includes:

- **Churn** — customers who left within the last month
- **Services** — phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- **Account info** — tenure, contract, payment method, paperless billing, monthly charges, and total charges
- **Demographics** — gender, age range, partner, and dependents

**Inspiration:** Explore churn prediction models and learn more about customer retention.

**Updated source:** A newer version is available from [IBM Community](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

## Local setup

1. Ensure conda (Miniconda or Anaconda) is installed.
2. Create and activate a conda environment named `customer-churn-prediction-mlops` with Python 3.12:

```bash
conda create -n customer-churn-prediction-mlops python=3.12 -y
conda activate customer-churn-prediction-mlops
pip install -r requirements.txt
```

> **Note:** Be careful when running or changing code that affects production. Use this local environment for development and testing only.
