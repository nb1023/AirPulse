import streamlit as st
import requests
import joblib
import pickle
import base64
import pandas as pd
import matplotlib.pyplot as plt

from analysis import process_pollutant,monthly_analysis
# Load the trained Random Forest models and imputers for different cities
city_models = {
    "R.K. Puram, Delhi, Delhi, India": {
        "model": joblib.load('PKL files/Delhi_random_forest_model.pkl'),
        "imputer": joblib.load('PKL files/imputer.pkl')
    },
    "Sanjay Nagar, Ghaziabad, India": {
        "model": joblib.load('PKL files/Ghaziabad_random_forest_model.pkl'),
        "imputer": joblib.load('PKL files/imputer.pkl')
    },
    "Knowledge Park - III, Greater Noida, India": {
        "model": joblib.load('PKL files/GNoida_random_forest_model.pkl'),
        "imputer": joblib.load('PKL files/imputer.pkl')
    },
    "Talkatora, Lucknow, India": {
        "model": joblib.load('PKL files/Lucknow_random_forest_model.pkl'),
        "imputer": joblib.load('PKL files/imputer.pkl')
    },
    # Add more cities as needed
}

# Replace this with your OpenAQ API key
api_key = "9122142749f2d354a43af188bc4486a59f678eed"

# Create session state variables
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

if "aqi" not in st.session_state:
    st.session_state.aqi = None

# Sidebar with radio buttons for page selection
page = st.sidebar.radio("Select Page", ["Home", "AQI Prediction", "Health Prediction","Analysis","News"])

if page == "Home":
    # # Set background image using CSS
    # st.markdown(
    #     """
    #     <style>
    #         body {
    #             background-image: url('delhi.jpg');
    #             background-size: cover;
    #             background-repeat: no-repeat;
    #         }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
            unsafe_allow_html=True
        )


    add_bg_from_local('bg.jpg')

    # Home Page Content

    st.title("Welcome to AirPulse💓")

    st.markdown("""
    This app allows you to predict the Air Quality Index (AQI) for different cities and provides insights into the potential health consequences based on the predicted AQI.
    """)


    st.markdown("""
    ## How to Use the App

    1. **Navigate to "AQI Prediction":** Click on the "AQI Prediction" page in the sidebar to predict the AQI for a selected city.

    2. **Select City:** Choose a city from the dropdown menu.

    3. **Predict AQI:** Click the "Predict AQI" button to fetch real-time air quality data and predict the AQI using a machine learning model.

    4. **Explore Results:** View the predicted AQI, check if the air quality is healthy or unhealthy, and explore the individual air quality parameters.

    5. **Visit "Health Predictions": Navigate to the 'Health Prediction' section to delve into forecasts related to population health, specifically focusing on the selected city's Air Quality Index (AQI). Discover insights into predictions associated with diseases such as Cancer and Respiratory ailments for a comprehensive understanding of potential health outcomes.
    
    5. **Visit "Analysis":** Check out the "Health Prediction" to explore Analysis based on Yearwise and Monthwise.

    Enjoy exploring air quality predictions and their potential health impacts!
    """)


elif page == "AQI Prediction":
    st.title("Air Quality Prediction")

    # Dropdown menu for selecting cities
    selected_city = st.selectbox("Select city", list(city_models.keys()))

    # Update the background image based on the selected city

    # Add more cities as needed

    # Update the background image using CSS
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-repeat: no-repeat;
            background-size: 100% 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

    if selected_city == "R.K. Puram, Delhi, Delhi, India":
        add_bg_from_local('delhi.jpg')
    elif selected_city == "Sanjay Nagar, Ghaziabad, India":
        add_bg_from_local('ghaziabad.jpg')
    elif selected_city == "Knowledge Park - III, Greater Noida, India":
        add_bg_from_local('noida.jpg')
    elif selected_city == "Talkatora, Lucknow, India":
        add_bg_from_local('lucknow.jpg')

    # Store selected city in session state
    st.session_state.selected_city = selected_city

    if st.button("Predict AQI"):
        # Fetch air quality data using OpenAQ API
        url = f"https://api.waqi.info/feed/{selected_city}/?token={api_key}"
        response = requests.get(url)
        json_data = response.json()

        if 'data' in json_data:
            data = json_data['data']
            aqi = data.get('aqi', 'N/A')

            # Store AQI in session state
            st.session_state.aqi = aqi

            pm25 = data['iaqi'].get('pm25', {}).get('v', 'N/A')
            pm10 = data['iaqi'].get('pm10', {}).get('v', 'N/A')
            o3 = data['iaqi'].get('o3', {}).get('v', 'N/A')
            no2 = data['iaqi'].get('no2', {}).get('v', 'N/A')
            so2 = data['iaqi'].get('so2', {}).get('v', 'N/A')
            co = data['iaqi'].get('co', {}).get('v', 'N/A')

            # Check for 'N/A' values
            if 'N/A' in [pm25, pm10, o3, no2, so2, co]:
                st.error("Some air quality parameters are not available.")
            else:
                # Create a dictionary with the retrieved air quality parameters
                input_data = {
                    'pm25': [float(pm25)],
                    'pm10': [float(pm10)],
                    'o3': [float(o3)],
                    'no2': [float(no2)],
                    'so2': [float(so2)],
                    'co': [float(co)]
                }

                # Convert the input data to a DataFrame
                input_df = pd.DataFrame(input_data)

                # Impute missing values with the mean using the loaded imputer
                imputer = city_models[selected_city]["imputer"]
                input_imputed = imputer.transform(input_df)

                # Predict AQI using the Random Forest model
                rf_model = city_models[selected_city]["model"]
                rf_aqi = rf_model.predict(input_imputed)

                st.write(f"The AQI in {selected_city} is: {rf_aqi[0]}")

                # Determine the air quality category and set color
                if rf_aqi[0] <= 50:
                    quality_message = "<strong>GOOD.</strong>"
                    color = "green"
                elif 51 <= rf_aqi[0] <= 100:
                    quality_message = "<strong>MODERATE.</strong>"
                    color = "yellow"
                elif 101 <= rf_aqi[0] <= 150:
                    quality_message = "<strong>UNHEALTHY FOR SENSITIVE GROUPS.</strong>"
                    color = "dark yellow"
                elif 151 <= rf_aqi[0] <= 200:
                    quality_message = "<strong>UNHEALTHY.</strong>"
                    color = "red"
                elif 201 <= rf_aqi[0] <= 300:
                    quality_message = "<strong>VERY UNHEALTHY.</strong>"
                    color = "orange"
                else:
                    quality_message = "<strong>HAZARDOUS.</strong>"
                    color = "blue"

                # Combine the message and quality_message into a single string
                styled_message = f'<p style="font-size: larger;">The air quality is <span style="color:{color}; font-size: larger;">{quality_message}</span></p>'

                st.markdown(styled_message, unsafe_allow_html=True)

                st.write("Parameters:")
                st.write(f"PM2.5: {pm25}")
                st.write(f"PM10: {pm10}")
                st.write(f"Ozone: {o3}")
                st.write(f"Nitrogen Dioxide: {no2}")
                st.write(f"Sulfur Dioxide: {so2}")
                st.write(f"Carbon Monoxide: {co}")

elif page == "Health Prediction":
    st.title("Health Prediction")
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


    add_bg_from_local('bg3.jpg')

    # Use selected city and AQI from session state
    st.write(f"Selected City: {st.session_state.selected_city}")
    st.write(f"Current AQI: {st.session_state.aqi}")

    # Load the trained models from pickle files
    with open('PKL files/model_health.pkl', 'rb') as file:
        model_health = pickle.load(file)

    with open('PKL files/model_general.pkl', 'rb') as file:
        model_general = pickle.load(file)

    with open('PKL files/model_vulnerable.pkl', 'rb') as file:
        model_vulnerable = pickle.load(file)

    # Load the trained models from the pickle file
    with open('PKL files/Disease.pkl', 'rb') as file:
        models = pickle.load(file)

    health_prediction = model_health.predict([[st.session_state.aqi]])
    st.write(f'Predicted for Overall Population Consequences: {health_prediction[0]}')
    # Radio button for population selection
    population_selection = st.radio("Select Population", ["General Population", "Vulnerable Population"])

    # Make predictions based on population selection
    if population_selection == "General Population":
        general_prediction = model_general.predict([[st.session_state.aqi]])
        st.write(f'Predicted General Population Consequences: {general_prediction[0]}')

    elif population_selection == "Vulnerable Population":
        vulnerable_prediction = model_vulnerable.predict([[st.session_state.aqi]])
        st.write(f'Predicted Vulnerable Population Consequences: {vulnerable_prediction[0]}')


        def make_predictions(aqi, disease):
            input_data = pd.DataFrame({'AQI': [aqi], 'Disease': [disease]})
            predictions = {col: model.predict(input_data) for col, model in models.items()}
            return predictions

        # Input form
        float_aqi = st.session_state.aqi

        # Convert float to integer
        aqi = int(float_aqi)

        disease = st.selectbox("Disease:", ['Cancer', 'Respiratory'])

        # Make predictions on button click
        if st.button("Predict"):
            predictions = make_predictions(aqi, 0 if disease == 'Cancer' else 1)
            for col, pred_value in predictions.items():
                st.write(f"{col}: {pred_value[0]}")





if page == "Analysis":
    st.title("Analysis")
    st.write(f"Selected City: {st.session_state.selected_city}")



    # Radio button for selecting pollutants
    selected_pollutant = st.radio("Select Pollutant", ["pm25", "pm10", "o3", "no2", "so2", "co", "AQI"])

    # Radio buttons for selecting analysis type
    selected_analysis = st.radio("Select Analysis Type", ["Year-wise Analysis", "Month-wise Analysis"])

    if selected_analysis == "Year-wise Analysis":
        if st.session_state.selected_city == "R.K. Puram, Delhi, Delhi, India":
            # Retrieve the DataFrame from the CSV file or any other source
            file_path = "CSV files/r.k.-puram, delhi, delhi, india-air-quality.csv"
            df = pd.read_csv(file_path)
            plt = process_pollutant(df, selected_pollutant, selected_pollutant.upper())
            st.pyplot(plt)

        elif st.session_state.selected_city == "Sanjay Nagar, Ghaziabad, India":
            # Retrieve the DataFrame from the CSV file or any other source
            file_path = "CSV files/sanjay-nagar, ghaziabad, india-air-quality.csv"
            df = pd.read_csv(file_path)
            plt = process_pollutant(df, selected_pollutant, selected_pollutant.upper())
            st.pyplot(plt)

        elif st.session_state.selected_city == "Knowledge Park - III, Greater Noida, India":
            # Retrieve the DataFrame from the CSV file or any other source
            file_path = "CSV files/knowledge-park - iii, greater noida, india-air-quality.csv"
            df = pd.read_csv(file_path)
            plt = process_pollutant(df, selected_pollutant, selected_pollutant.upper())
            st.pyplot(plt)

    # Add similar conditions for Month-wise and Day-wise Analysis
    elif selected_analysis == "Month-wise Analysis":
        if st.session_state.selected_city == "R.K. Puram, Delhi, Delhi, India":
            # Retrieve the DataFrame from the CSV file or any other source
            file_path = "CSV files/r.k.-puram, delhi, delhi, india-air-quality.csv"
            df = pd.read_csv(file_path)
            monthly_analysis(df, selected_pollutant)
            st.pyplot(plt)

        elif st.session_state.selected_city == "Sanjay Nagar, Ghaziabad, India":
            file_path = "CSV files/sanjay-nagar, ghaziabad, india-air-quality.csv"
            df = pd.read_csv(file_path)
            monthly_analysis(df, selected_pollutant)
            st.pyplot(plt)

        elif st.session_state.selected_city == "Knowledge Park - III, Greater Noida, India":
            file_path = "CSV files/knowledge-park - iii, greater noida, india-air-quality.csv"
            df = pd.read_csv(file_path)
            monthly_analysis(df, selected_pollutant)
            st.pyplot(plt)

elif page == "News":
    import streamlit as st
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager


    # Hide Streamlit style
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Pollution News")

    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = webdriver.Chrome(service=service, options=options)

    driver.get('https://www.indiatvnews.com/topic/weather')

    news_items = driver.find_elements(By.CSS_SELECTOR, "div.row.newsListBox")

    news = []

    for item in news_items:
        headline = item.find_elements(By.CSS_SELECTOR, "h3.title")
        newsdesc = item.find_elements(By.CSS_SELECTOR, "p.dic")
        daydatetime = item.find_elements(By.CSS_SELECTOR, "span.deskTime")
        image = item.find_elements(By.CSS_SELECTOR, "img")
        news_url = item.find_elements(By.CSS_SELECTOR, "a.thumb")
        for i in range(len(headline)):
            news.append({"headline": headline[i].text,
                         "newsdesc": newsdesc[i].text,
                         "daydatetime": daydatetime[i].text,
                         "image": image[i].get_attribute("data-original"),
                         "news_url": news_url[i].get_attribute("href")})

    # Close the Chrome WebDriver
    driver.quit()

    # Display the news items as a table or list using Streamlit's "st" functions
    if len(news) == 0:
        st.warning("No news found.")
    else:
        for i, item in enumerate(news):
            st.write(f"**{i + 1}. {item['headline']}**")
            st.write(f"{item['newsdesc']}")
            st.write(f"Published: {item['daydatetime']}")
            st.image(item['image'])
            st.write(f"URL: {item['news_url']}")
            st.write("")












