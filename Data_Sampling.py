import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('processed_loan_data.csv')

# Feature selection
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handling imbalanced data
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
