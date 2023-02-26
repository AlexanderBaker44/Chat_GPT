import streamlit as st
import pandas as pd
import pickle
import statsmodels.api as sm

# Load the pre-trained SARIMAX model
with open('sarimax_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CPI data
cpi_data = pd.read_csv('cpi.csv', parse_dates=['DATE'], index_col=['DATE'])

# Define the Streamlit app
st.title('CPI Forecasting App')

# Create a sidebar with a slider to select the number of periods to forecast
num_periods = st.sidebar.slider('Select the number of periods to forecast:', 1, 24)

# Make the forecast using the SARIMAX model
forecast = model.forecast(steps=num_periods)

# Define the date range for the forecast
last_date = cpi_data.index[-1]
forecast_range = pd.date_range(last_date, periods=num_periods+1, freq='MS')[1:]

# Combine the forecast with the existing CPI data
combined_data = pd.concat([cpi_data, pd.DataFrame({'CPI': forecast}, index=forecast_range)])

# Display the forecasted CPI values in a line chart
st.line_chart(combined_data['CPI'])
