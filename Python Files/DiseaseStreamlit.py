import streamlit as st
import pandas as pd
import pickle

# Load the trained models from the pickle file
with open('../PKL files/Disease.pkl', 'rb') as file:
    models = pickle.load(file)


def make_predictions(aqi, disease):
    input_data = pd.DataFrame({'AQI': [aqi], 'Disease': [disease]})
    predictions = {col: model.predict(input_data) for col, model in models.items()}
    return predictions


# Streamlit app
def main():
    st.title("Disease Prediction App")

    # Input form
    aqi = st.slider("Air Quality Index (AQI):", min_value=0, max_value=500, value=100)
    disease = st.selectbox("Disease:", ['Cancer', 'Respiratory'])

    # Make predictions on button click
    if st.button("Predict"):
        predictions = make_predictions(aqi, 0 if disease == 'Cancer' else 1)
        st.subheader("Predicted values:")
        for col, pred_value in predictions.items():
            st.write(f"{col}: {pred_value[0]}")


if __name__ == '__main__':
    main()
