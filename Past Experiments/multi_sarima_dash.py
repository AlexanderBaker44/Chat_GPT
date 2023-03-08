
import pickle
from typing import Dict
import pandas as pd
from pmdarima.arima import auto_arima

def forecast_auto_arima(file_names: list, periods: int) -> Dict[str, pd.DataFrame]:
    forecasts = {}
    for file_name in file_names:
        with open(file_name, 'rb') as pkl:
            model = pickle.load(pkl)
        forecast = model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)
        forecast_df = pd.DataFrame(forecast[0], columns=['forecast'])
        forecast_df['lower'] = forecast[1][:, 0]
        forecast_df['upper'] = forecast[1][:, 1]
        forecasts[file_name] = forecast_df
    return forecasts


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title='Auto-ARIMA Forecasts', layout='wide')

# Define the file names of the pickled auto_arima models
file_names = ['sarima_cpaltt01usm657n.pickle',
              'sarima_m2sl.pickle',
              'sarima_ppiaco.pickle',
              'sarima_unrate.pickle']

# Display a slider widget for selecting the number of periods
num_periods = st.slider('Select the number of periods to forecast', min_value=1, max_value=24, value=12)


# Generate the forecasts
forecasts = forecast_auto_arima(file_names, num_periods)

# Plot the forecasts
for file_name, forecast_df in forecasts.items():
    fig, ax = plt.subplots()
    ax.plot(forecast_df['forecast'], label='Forecast')
    ax.fill_between(forecast_df.index, forecast_df['lower'], forecast_df['upper'], alpha=0.3)
    ax.set_title(file_name)
    ax.legend()
    st.pyplot(fig)
