# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('../CSV files/Health.csv')

# Extract the features (AQI) and target variables
X = df[['AQI']]
y_health = df['Health_Consequences']
y_general = df['General_Population']
y_vulnerable = df['Vulnerable_Population']

# Split the data into training and testing sets
X_train, X_test, y_health_train, y_health_test, y_general_train, y_general_test, y_vulnerable_train, y_vulnerable_test = train_test_split(
    X, y_health, y_general, y_vulnerable, test_size=0.2, random_state=42
)

# Create decision tree classifiers
model_health = DecisionTreeClassifier()
model_general = DecisionTreeClassifier()
model_vulnerable = DecisionTreeClassifier()

# Train the models
model_health.fit(X_train, y_health_train)
model_general.fit(X_train, y_general_train)
model_vulnerable.fit(X_train, y_vulnerable_train)

# Save the models to pickle files
with open('../PKL files/model_health.pkl', 'wb') as file:
    pickle.dump(model_health, file)

with open('../PKL files/model_general.pkl', 'wb') as file:
    pickle.dump(model_general, file)

with open('../PKL files/model_vulnerable.pkl', 'wb') as file:
    pickle.dump(model_vulnerable, file)

# Print the accuracies (optional)
health_predictions = model_health.predict(X_test)
general_predictions = model_general.predict(X_test)
vulnerable_predictions = model_vulnerable.predict(X_test)

accuracy_health = accuracy_score(y_health_test, health_predictions)
accuracy_general = accuracy_score(y_general_test, general_predictions)
accuracy_vulnerable = accuracy_score(y_vulnerable_test, vulnerable_predictions)

print(f'Accuracy (Health): {accuracy_health}')
print(f'Accuracy (General Population): {accuracy_general}')
print(f'Accuracy (Vulnerable Population): {accuracy_vulnerable}')
