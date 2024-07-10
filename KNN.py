import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from Data_Sampling import X_test, X_train, y_train, y_test

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can choose different values for n_neighbors

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print performance metrics
print("Confusion Matrix Training:\n", confusion_matrix(y_train, y_train_pred))
print("Training Accuracy:", train_accuracy)
print("Confusion Matrix Testing:\n", confusion_matrix(y_test, y_test_pred))
print("Testing Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Hyperparameter tuning: You can use GridSearchCV to find the optimal n_neighbors
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_gscv.fit(X_train, y_train)

# Best parameters and score
print("Best parameters found: ", knn_gscv.best_params_)
print("Best cross-validation accuracy: ", knn_gscv.best_score_)

# Evaluate the best model on the test set
best_knn_model = knn_gscv.best_estimator_
y_test_pred_best = best_knn_model.predict(X_test)

# Print the performance of the best model
print("Best KNN Model Performance on Test Set:")
print("Confusion Matrix Testing:\n", confusion_matrix(y_test, y_test_pred_best))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_best))
print("Classification Report:\n", classification_report(y_test, y_test_pred_best))