# Credit Analysis with PySpark ML

This project implements machine learning models using PySpark ML for credit analysis. 
The goal is to predict the optimal credit amount and assess the creditworthiness of clients based on various features.
1. Predicting the credit amount
2. Assessing creditworthiness (probability of loan repayment)

## Requirements

- Python 3.10.11
- PySpark 3.1.2
- NumPy
- Pandas
- scikit-learn
- SciPy

## Structure of the Project

- `predict_credit_amount.py` - model for predicting the optimal credit amount
- `predict_credit_score.py` - model for assessing creditworthiness
- `data/` - folder containing datasets
  - `predict_credit_amount/` - data for the credit amount prediction model
  - `predict_credit_score/` - data for the credit score prediction model

## Data Schema

Each dataset has the following schema:

| Field Name                | Type   | Description                                     |
|---------------------------|--------|-------------------------------------------------|
| client_id                 | uuid   | identifier of the client                        |
| age                       | int    | age of the client                               |
| sex                       | string | sex of the client                               |
| married                   | string | marital status of the client                    |
| salary                    | float  | monthly salary of the client (₽)                |
| successful_credit_history | int    | number of successfully closed credits           |
| credit_completed_amount   | float  | total amount of successfully closed credits (₽) |
| active_credits            | int    | number of active credits                        |
| active_credits_amount     | float  | total amount of active credits (₽)              |
| credit_amount             | float  | requested credit amount (₽)                     |




## Models

### Predicting the optimal credit amount

Model uses Random Forest Regressor to predict the optimal credit amount based on the following features
- age
- sex
- married
- salary
- successful_credit_history
- credit_completed_amount
- active_credits
- credit_amount

Sample output:
```
    RMSE: 32754.12
    Model maxDepth: 5
    Model maxBins: 32
```

### Assessing creditworthiness

Model uses Random Forest Classifier to predict the probability of successful loan repayment based on the following features
- age
- sex
- married
- salary
- successful_credit_history
- credit_completed_amount
- active_credits
- credit_amount

Sample output:
```
    Accuracy: 0.842
    Model maxDepth: 7
    Model maxBins: 64

```

## Dependency Installation

1. Clone the repository:
   ```bash
   git clone
   ```
   
2. Install the required dependencies:
   ```bash
   pipenv sync
   ```
3. Run the models:
   ```bash
   python3 predict_credit_amount.py
   python3 predict_credit_score.py
   ```