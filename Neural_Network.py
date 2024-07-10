import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib
from Data_Sampling import X_test, X_train, y_train, y_test

# Set the Matplotlib backend
matplotlib.use('TkAgg')

# Train initial neural network model
nnetmodel = MLPClassifier(solver='adam', hidden_layer_sizes=(17,), random_state=0, max_iter=2000, alpha=0.0001, learning_rate='adaptive', batch_size=32)
nnetmodel.fit(X_train, y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(y_train, Y_train_pred)
print("Accuracy Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(y_test, Y_test_pred)
print("Accuracy Test:", accte)

# Hyperparameter tuning: Evaluating different numbers of hidden neurons
accuracies = np.zeros((3, 20), float)
for k in range(0, 20):
    nnetmodel = MLPClassifier(solver='adam', hidden_layer_sizes=(k+1,), random_state=0, max_iter=2000, alpha=0.0001, learning_rate='adaptive', batch_size=32)
    nnetmodel.fit(X_train, y_train)
    Y_train_pred = nnetmodel.predict(X_train)
    acctr = accuracy_score(y_train, Y_train_pred)
    accuracies[1, k] = acctr
    Y_test_pred = nnetmodel.predict(X_test)
    accte = accuracy_score(y_test, Y_test_pred)
    accuracies[2, k] = accte
    accuracies[0, k] = k+1

plt.plot(range(1, 21), accuracies[1, :], label='Training Accuracy')
plt.plot(range(1, 21), accuracies[2, :], label='Testing Accuracy')
plt.xlim(1, 20)
plt.xticks(range(1, 21))
plt.xlabel('Hidden Neurons')
plt.ylabel('Accuracy')
plt.title('Neural Network')
plt.legend()
plt.show()

headers = ["Hidden Neurons", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)
maxi = np.array(np.where(accuracies == accuracies[2:].max()))
table = tabulate(accuracies[:, maxi[1, :]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

# Train final neural network model with the best number of hidden neurons
nnetmodel = MLPClassifier(solver='adam', hidden_layer_sizes=(11,), random_state=0, max_iter=2000, alpha=0.0001, learning_rate='adaptive', batch_size=32)
nnetmodel.fit(X_train, y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(y_train, Y_train_pred)
print("Accuracy Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(y_test, Y_test_pred)
print("Accuracy Test:", accte)

# Document the results
report = pd.DataFrame(columns=['Model', 'Training Accuracy', 'Testing Accuracy'])
report.loc[len(report)] = ['Neural Network', acctr, accte]
print("\n################")
print("# Final Report #")
print("################\n")
print(report)