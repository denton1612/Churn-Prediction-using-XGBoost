# Bank Customer Data for Churn Analysis and Prediction

## Project Description:
This notebook focuses on predicting customer churn — whether a bank customer will leave the bank (Exited = 1) or remain (Exited = 0).
The model is trained using demographic, financial, and behavioral customer attributes provided in the dataset.

The main goal is to build a high-performance churn prediction model using LightGBM, optimized with Optuna to maximize the ROC-AUC score.

## Project Workflow Overview:
### 1. Data Loading & Preprocessing:
Utility functions are used to handle loading and preparing the dataset:
- Load .csv files
- Remove irrelevant features (e.g., CustomerId, Surname, id)
- Separate input (X) and target (y)
- Encode categorical features (Gender, Geography)
- Add additional engineered features (e.g., HasBalance)
- Handle missing values
- Mark categorical features for LightGBM

### 2. Hyperparameter Optimization (Optuna):
Optuna is used to search for the best LightGBM hyperparameters:
- The search objective is maximize ROC-AUC
- 5-fold stratified cross-validation is used to evaluate performance
- The best hyperparameters and AUC score are displayed after training

### 3. Model Training (LightGBM):
Using the best hyperparameters found earlier:
- A final LightGBM classifier is built
- The model is trained on the full training dataset
- Categorical features are passed explicitly for better performance

### 4. Prediction on Test Set:
- Preprocessing applied to the train set is replicated on the test set
- Predictions are generated using model.predict_proba
- A submission.csv file is generated with:

## Technologies Used:
| Type                     | Libraries / Tools              |
| ------------------------ | ------------------------------ |
| Data processing          | pandas, numpy                  |
| Encoding / preprocessing | sklearn                        |
| Model                    | LightGBM                       |
| Hyperparameter tuning    | Optuna                         |
| Evaluation               | ROC-AUC score, StratifiedKFold |
