# Credit Risk Assessment and Loan Prediction

![Credit Risk Assessment and Loan Prediction](Risk.jpeg)

## Overview

This project involves analyzing a loan dataset to predict credit risk using loan status and loan amounts. The dataset contains information about borrowers, including age, income, home ownership, employment length, loan intent, and credit history. The analysis includes data preprocessing, exploratory data analysis, and the application of various machine learning models.

For credit risk analysis (classification), models like Decision Trees, Random Forest, Logistic Regression, K-Nearest Neighbors, Gradient Boosting, and XGBoost were used. XGBoost performed best with an accuracy of 93.7% and an F1 score of 0.839. For loan amount prediction (regression), Linear Regression, XGBoost, and Artificial Neural Networks (ANN) were employed. XGBoost outperformed others with an R² of 0.995 and the lowest RMSE of 425.22.

The process involved data cleaning, feature engineering, model training, hyperparameter tuning using GridSearchCV, and performance evaluation using metrics like accuracy, precision, recall, F1 score for classification, and RMSE, MSE, MAE, R² for regression. Feature importance analysis was also conducted to identify the most influential factors in predictions.

**Goal**

1. Credit Risk Classification: Categorizing loan applicants based on their creditworthiness to make informed decisions about loan approvals and interest rates.
2. Loan Amount Prediction: Predicting suitable loan amounts for approved applicants, balancing their financial needs with the risk of default.


## Business Understanding

### Stakeholders
Business stakeholders include:
- Financial Institutions (Banks, Credit Unions, Lending Companies)
- Loan Applicants (Customers)
- Credit Bureaus and Reporting Agencies
- Regulatory Bodies
- Investors and Shareholders of Financial Institutions
- Loan Officers and Risk Management Teams

### Key Business Questions

- 
- 
- 
- 

## Data Understanding and Analysis

### Source of Data

The data for this analysis was obtained from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download). It consists of 32,581 observations of 12 variables


### Description of Data

 - person_age: The age of the borrower when securing the loan.
 - person_income: The borrower’s annual earnings at the time of the loan.
 - person_home_ownership: Type of home ownership.
 - person_emp_length: The amount of time in years that the borrower is employed.
 - loan_intent: Loan purpose.
 - loan_grade: Classification system based on credit history, collateral quality, and likelihood of repayment.
    - A: The borrower has a high creditworthiness, indicating low risk.
    - B: The borrower is relatively low-risk, but not as creditworthy as Grade A.
    - C: The borrower’s creditworthiness is moderate.
    - D: The borrower is considered to have higher risk compared to previous grades.
    - E: The borrower’s creditworthiness is lower, indicating a higher risk.
    - F: The borrower poses a significant credit risk.
    - G: The borrower’s creditworthiness is the lowest, signifying the highest risk.
 - loan_amnt: Total amount of the loan.
 - loan_int_rate: Interest rate of the loan.
 - loan_status: Dummy variable indicating default (1) or non-default (0).
 A default occurs when a borrower is unable to make timely payments, misses payments, or avoids or stops making payments on interest or principal owed.
 - loan_percent_income: Ratio between the loan amount and the annual income.
 - cb_person_cred_hist_length: The number of years of personal history since the first loan taken.
 - cb_person_default_on_file: Indicates if the person has previously defaulted.

## Exploratory Data Analysis


#### Analyzing loan intent of borrowers:

- **Analysis**:  Educational purposes represent the highest loan intent percentage at 19.86%. This suggests that a significant portion of borrowers are investing in their education, possibly to further their careers or pursue higher levels of education.On the other hand, home improvement purposes represent the lowest percentage at 11.08%, indicating a smaller but still notable portion are investing in renovating or upgrading their homes..

#### 2. Analyzing Default on File Distribution:

- **Analysis**: 82% of borrowers have a history of defaults on their loans, suggesting that there is a prevalent trend of financial difficulties among borrowers. Conversely, 18% of borrowers stand out for their clean repayment records, indicating a minority who have managed to navigate their financial obligations successfully.

#### 3. Analyzing Loan Intent by age group:

- **Analysis**: Borrowers aged 21-30 are the largest demographic securing loans, with education being the predominant reason and home improvement ranking lowest. This implies a focus on investing in education and potentially early career development among younger borrowers. For borrowers aged 31-40 and 41-50, the primary reason shifts towards medical needs, indicating a growing focus on healthcare in this age group. Finally, for borrowers aged 51-60 and 61-70, personal reasons become the main motivation for loans, reflecting a diverse range of financial needs or aspirations among older individuals. 

#### 4. Analyzing Default Status by Age Group:

- **Analysis**: younger individuals (21-30) are more represented in this dataset, with a much higher count of individuals either not taking loans or defaulting (loan status 0), while fewer people successfully manage their loans (loan status 1)


## Modeling
