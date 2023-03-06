import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load pickled VAR model
with open('var_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
data = pd.read_csv('fred_data.csv', index_col=0)

num_steps = st.slider('Select number of steps to forecast:', 1, 24, 12)

forecast = model.forecast(data.iloc[-24:].values, num_steps)

pd.DataFrame(forecast).plot(kind='line')
st.pyplot()
print(forecast)
# Run app
