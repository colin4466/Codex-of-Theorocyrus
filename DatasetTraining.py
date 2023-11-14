import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('your-dataset.csv')  # Replace 'your_dataset.csv' with the actual file path

# Select relevant features (latitude, longitude, and temperature)
features = data[['LATITUDE', 'LONGITUDE', 'TMAX', 'TMIN', 'PRCP']]

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create a column for anomaly labels (1 for normal, -1 for anomaly)
data['ANOMALY_LABEL'] = 1  # Assume all data points are normal initially

# Identify anomalies based on specific conditions (you can customize this)
anomaly_condition = (data['TMAX'] > 100) | (data['TMIN'] < -50) | (data['PRCP'] > 1000)
data.loc[anomaly_condition, 'ANOMALY_LABEL'] = -1

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination based on your dataset
model.fit(features_standardized)

# Predict anomalies on the training set
train_data['PREDICTION'] = model.predict(scaler.transform(train_data[['LATITUDE', 'LONGITUDE', 'TMAX', 'TMIN', 'PRCP']]))

# Evaluate the model on the training set
print("Training Set Classification Report:")
print(classification_report(train_data['ANOMALY_LABEL'], train_data['PREDICTION']))

# Predict anomalies on the test set
test_data['PREDICTION'] = model.predict(scaler.transform(test_data[['LATITUDE', 'LONGITUDE', 'TMAX', 'TMIN', 'PRCP']]))

# Evaluate the model on the test set
print("\nTest Set Classification Report:")
print(classification_report(test_data['ANOMALY_LABEL'], test_data['PREDICTION'], zero_division=1))
