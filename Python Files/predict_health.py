import pickle

# Load the trained models from pickle files
with open('../PKL files/model_health.pkl', 'rb') as file:
    model_health = pickle.load(file)

with open('../PKL files/model_general.pkl', 'rb') as file:
    model_general = pickle.load(file)

with open('../PKL files/model_vulnerable.pkl', 'rb') as file:
    model_vulnerable = pickle.load(file)

# Take input from the user (replace this with your input method)
new_AQI = float(input("Enter the AQI value: "))
new_AQI = [[new_AQI]]

# Make predictions
health_prediction = model_health.predict(new_AQI)
general_prediction = model_general.predict(new_AQI)
vulnerable_prediction = model_vulnerable.predict(new_AQI)

# Display predictions
print(f'Predicted Health Consequences: {health_prediction[0]}')
print(f'Predicted General Population: {general_prediction[0]}')
print(f'Predicted Vulnerable Population: {vulnerable_prediction[0]}')
