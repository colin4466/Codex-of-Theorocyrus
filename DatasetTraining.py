import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_data = pd.read_csv('your-dataset.csv')

# Replace erroneous values (999.9) with NaN in the training data
train_data['HPCP'].replace(999.9, np.nan, inplace=True)
train_data.dropna(subset=['HPCP'], inplace=True)

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Replace erroneous values (999.9) with NaN in the test data
test_data['HPCP'].replace(999.9, np.nan, inplace=True)
test_data.dropna(subset=['HPCP'], inplace=True)

# Define features and target variable for training
X_train = train_data[['HPCP']]
y_train = train_data['ONGOING_FLOOD']

# Models to use
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train and predict using each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict likelihood of flood in test data
    X_test = test_data[['HPCP']]
    predictions = model.predict(X_test)

    # Add predictions to the test data
    test_data[f'Predicted_Flood_{name}'] = predictions

# Display the predictions for each model
for name in models.keys():
    print(f'Predictions from {name}:')
    print(test_data[f'Predicted_Flood_{name}'])
    print('\n')
