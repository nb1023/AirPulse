import requests
import pandas as pd
import joblib
import numpy as np


def aqi_checker(city_name):
    api_key = "9122142749f2d354a43af188bc4486a59f678eed"
    url = f"https://api.waqi.info/feed/{city_name}/?token={api_key}"

    response = requests.get(url)
    json_data = response.json()

    if 'data' in json_data:
        data = json_data['data']
        aqi = data.get('aqi', 'N/A')
        index = data.get('idx', 'N/A')
        city = data['city'].get('name', 'N/A')
        Dominant_pollutant = data.get('dominentpol', 'N/A')
        co = data['iaqi'].get('co', {}).get('v', 'N/A')
        h = data['iaqi'].get('dew', {}).get('v', 'N/A')
        no2 = data['iaqi'].get('no2', {}).get('v', 'N/A')
        o3 = data['iaqi'].get('o3', {}).get('v', 'N/A')
        p = data['iaqi'].get('p', {}).get('v', 'N/A')
        pm25 = data['iaqi'].get('pm25', {}).get('v', 'N/A')
        pm10 = data['iaqi'].get('pm10', {}).get('v', 'N/A')
        so2 = data['iaqi'].get('so2', {}).get('v', 'N/A')
        tem = data['iaqi'].get('t', {}).get('v', 'N/A')
        w = data['iaqi'].get('w', {}).get('v', 'N/A')

        print("The aqi in", city_name, "is:", aqi)
        print("The Index in", city_name, "is:", index)
        print("Address:", city)
        print("Dominant Pollutant:", Dominant_pollutant)
        print("Carbon Monoxide:", co)
        print("Nitrogen Dioxide:", no2)
        print("Ozone:", o3)
        print("pm25:", pm25)
        print("pm25:", pm10)
        print("Sulphur Dioxide:", so2)

        # Load the trained Random Forest model from the file
        rf_model = joblib.load('Faridabad_random_forest_model.pkl')

        # Load the SimpleImputer with the mean strategy
        imputer = joblib.load('../PKL files/imputer.pkl')

        # Create a dictionary with the retrieved air quality parameters
        input_data = {
            ' pm25': [pm25],
            ' pm10': [pm10],
            ' o3': [o3],
            ' no2': [no2],
            ' so2': [so2],
            ' co': [co]
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Impute missing values with the mean using the loaded imputer
        input_imputed = imputer.transform(input_df)

        # Predict AQI using the Random Forest model
        rf_aqi = rf_model.predict(input_imputed)

        # Print the predicted AQI using the Random Forest model
        print("Predicted AQI (Random Forest) for", city_name, "is:", rf_aqi[0])



    else:
        print("No data available for this city:", city_name)


aqi_checker('DITE Okhla, Delhi, Delhi, India')





