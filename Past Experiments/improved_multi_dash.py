import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Set page title
st.set_page_config(page_title='Fred Data', layout='wide')
models = []

with open('sarima_cpaltt01usm657n.pickle', 'rb') as f:
    models.append(pickle.load(f))

with open('sarima_m2sl.pickle', 'rb') as f:
    models.append(pickle.load(f))

with open('sarima_ppiaco.pickle', 'rb') as f:
    models.append(pickle.load(f))

with open('sarima_unrate.pickle', 'rb') as f:
    models.append(pickle.load(f))

# Read in data
df = pd.read_csv('fred_data_standardized.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
def make_predictions(model, n):
    """
    Makes n predictions using a trained auto_arima model.

    Args:
        model (AutoARIMA): A trained auto_arima model.
        n (int): Number of steps to forecast.

    Returns:
        numpy.ndarray: An array of n predicted values.
    """
    return model.predict(n_periods=n)
n_steps = st.sidebar.slider("Select the number of steps to forecast", 1, 24, 12, 1)

predictions = []

# iterate through each model
for model in models:
    # make predictions with n_steps
    pred = make_predictions(model, n_steps)
    # extract only the predicted values
    values = pred.values.ravel()
    # append to the predictions list
    predictions.append(values)
# Concatenate the predicted dataframes into one dataset
print(predictions)

df_predictions = pd.DataFrame(predictions)
df_predictions_transposed = df_predictions.T
df_predictions_transposed.columns = ['CPALTT01USM657N', 'UNRATE', 'PPIACO', 'M2SL']
print(df_predictions_transposed)
start_date = '2023-01-01'
index = pd.date_range(start_date, periods=n_steps, freq='MS')

df_predictions_transposed.index = index
df_predictions_transposed = df_predictions_transposed.reset_index()
df_predictions_transposed = df_predictions_transposed.rename(columns={'index': 'DATE'})
# Loop through each column in the DataFrame and plot on separate line chart
print(df_predictions_transposed)
df = pd.concat([df, df_predictions_transposed])
print(df)
for col in df.columns[1:]:
    fig, ax = plt.subplots()
    ax.plot(df['DATE'], df[col])
    ax.set(title=col, xlabel='Date', ylabel='Value')
    st.pyplot(fig)
