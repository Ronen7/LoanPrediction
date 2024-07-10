#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:03:13 2024

@author: ronen
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('LoanPrediction.csv')

data = data.drop('Loan_ID', axis=1)

# Determine the percentage of missing values in each column
missing_percentage = data.isnull().sum() / len(data) * 100

# Drop columns with more than 5% missing values
data = data.drop(columns=missing_percentage[missing_percentage > 5].index)


# Determine the percentage of missing values in each row
row_missing_percentage = data.isnull().sum(axis=1) / data.shape[1] * 100

# Drop rows with more than 10% missing values
data = data.drop(index=row_missing_percentage[row_missing_percentage > 10].index)

# Calculate the total percentage of missing values in the entire dataset
total_missing = data.isnull().sum().sum()
total_values = data.size
total_missing_percentage = (total_missing / total_values) * 100
print(total_missing)
# Print the total percentage of missing values
print(f"Total percentage of missing values in the dataset: {total_missing_percentage:.2f}%")

# Impute remaining missing values
# Fill numeric columns with mean, except Dependents and Loan_Amount_Term
data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = data[['ApplicantIncome',	'CoapplicantIncome', 'LoanAmount']].fillna(round(data[['ApplicantIncome',	'CoapplicantIncome', 'LoanAmount']].mean(),0))

# Fill Dependents with mode
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])

# Fill Loan_Amount_Term with mode
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])

# Fill Gender with mode
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
# Fill Property Area with mode
data['Property_Area'] = data['Property_Area'].fillna(data['Property_Area'].mode()[0])

label_encoder = LabelEncoder()
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])

# Encoding categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education'], drop_first=True)

# Convert boolean columns to integers (0 and 1)
for col in data.select_dtypes(include=['bool']).columns:
    data[col] = data[col].astype(int)

data.to_csv('processed_loan_data.csv', index=False)
