import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_data = pd.read_csv('your_dataset.csv')

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

# Choose a model and train it
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict likelihood of flood in test data
X_test = test_data[['HPCP']]
predictions = model.predict(X_test)

# Add predictions to the test data
test_data['Predicted_Flood'] = predictions

# Visualize predicted floods
plt.figure(figsize=(10, 5))
plt.plot(test_data['DATE'], test_data['HPCP'], label='Precipitation', color='blue')
plt.scatter(test_data[test_data['Predicted_Flood'] == 1]['DATE'], test_data[test_data['Predicted_Flood'] == 1]['HPCP'], color='red', label='Predicted Flood')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.title('Predicted Flood based on Precipitation')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
