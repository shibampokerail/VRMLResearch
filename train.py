from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Your data preparation code here

data = pd.read_excel("student_data.xlsx")

data = data[
    ["Professors-Duration", "Students-Count", "Students-Duration", "Staff-Count", "Staff-Duration", "Webpage-Count",
     "Webpage-Duration", "Docs-Count", "Docs-Duration", "Video-Count", "Video-Duration", "Informed-Duration.1",
     "All-Success-Duration"]]

# Print rows with data not available
# rows_with_nan = data[data.isnull().any(axis=1)]
# print("Rows with NaN data:")
# print(rows_with_nan)

data = data.dropna()

# print(data)

predict = "All-Success-Duration"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Initialize 5-fold cross-validation
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100, random_state=42)  # Example model, replace with your chosen model

model.fit(x_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

for train_index, val_index in kf.split(x_train):
    X_fold_train, X_fold_val = x_train[train_index], x_train[val_index]

    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]


    # Train your machine learning model on X_train and y_train

    model.fit(X_fold_train, y_fold_train)

    y_val_pred = model.predict(X_fold_val)

    fold_accuracy = accuracy_score(y_fold_val, y_val_pred)

    cv_scores.append(fold_accuracy)


    # Evaluate the model's performance (e.g., accuracy)

    # Append the score to the scores list

average_cv_accuracy = sum(cv_scores) / len(cv_scores)

# Print the average score to assess model performance

y_test_pred = model.predict(x_test)

test_accuracy = accuracy_score(y_test, y_test_pred)


print("Cross-Validation Average Accuracy:", average_cv_accuracy)

print("Holdout Test Set Accuracy:", test_accuracy)