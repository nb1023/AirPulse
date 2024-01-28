import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read the CSV file into a DataFrame
df = pd.read_csv('../CSV files/Disease.csv')


# Convert 'Cancer' to 0 and 'Respiratory' to 1 in the 'Disease' column
df['Disease'] = df['Disease'].map({'Cancer': 0, 'Respiratory': 1})

# Define features (X) and target (y)
X = df[['AQI', 'Disease']]
y_columns = ['Stay Informed','Indoor Activities', 'Air Purifiers', 'Ventilation', 'Use Masks', 'Stay Hydrated']

# Initialize classifiers for each target column
models = {col: DecisionTreeClassifier() for col in y_columns}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df[y_columns], test_size=0.2, random_state=42)

# Fit each model to the corresponding target column
for col, model in models.items():
    model.fit(X_train, y_train[col])

# Make predictions on the test set for each target column
predictions = {col: model.predict(X_test) for col, model in models.items()}

# Evaluate the accuracy of each model
accuracies = {col: accuracy_score(y_test[col], predictions[col]) for col in y_columns}
print("Accuracies:", accuracies)

# Now you can use the trained models to make predictions for new data
new_data = pd.DataFrame({'AQI': [500], 'Disease': [1]})
predictions_new_data = {col: models[col].predict(new_data) for col in y_columns}
print("Predicted values:", predictions_new_data)

import pickle

# Save models to a pickle file
with open('models.pkl', 'wb') as file:
    pickle.dump(models, file)

