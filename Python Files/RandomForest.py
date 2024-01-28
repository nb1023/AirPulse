import pandas as pd
from sklearn.impute import SimpleImputer
import joblib  # Import joblib for model loading
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load your dataset from a CSV file
data = pd.read_csv('talkatora,-lucknow, india-air-quality.csv')

# Replace empty spaces with 'NaN'
data.replace(' ', 'NaN', inplace=True)

# Define the input parameters and the target variable
input_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
output_col = 'AQI'

# Extract input features and target variable
X = data[input_cols]
y = data[output_col]

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Specify the absolute path to save the trained imputer
imputer_filename = '../PKL files/imputer.pkl'
joblib.dump(imputer, imputer_filename)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize and fit the RandomForestRegressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Save the trained model to a pickle file
model_filename = 'random_forest_model.pkl'
joblib.dump(rf_model, model_filename)
