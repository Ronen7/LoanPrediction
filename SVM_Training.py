from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Data_Sampling import X_test, X_train, y_train, y_test

# Initialize the SVM model
svm_model = SVC(probability=True, random_state=42)

# Define the parameter grid for Grid Search
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Initialize the Grid Search with cross-validation
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found for SVM: ", grid_search_svm.best_params_)
print("Best cross-validation accuracy for SVM: ", grid_search_svm.best_score_)

# Evaluate the best SVM model on the test set
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)
print("SVM Performance after tuning:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm, pos_label='Y'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm, pos_label='Y'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm, pos_label='Y'):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, best_svm_model.predict_proba(X_test)[:, 1]):.4f}")
