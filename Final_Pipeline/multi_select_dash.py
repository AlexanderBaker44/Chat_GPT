import streamlit as st
import pandas as pd


def display_data():
    # read in the data
    data = pd.read_csv('data/fred_data_scaled_with_preds.csv', index_col=0)


    data = data.rename(columns={'CPALTT01USM657N': 'Adjusted_CPI'})

    st.header('Economic Factors Display')

    cols = st.multiselect("Select one or more columns", data.columns.tolist(), default=["CPI"])

    # Filter the data based on selected columns
    filtered_data = data[cols]

    # create select slider for start date
    start_date = st.select_slider('Select start date', options=data.index, value=data.index.min())

    # create select slider for end date
    end_date = st.select_slider('Select end date', options=data.index, value=data.index.max())

    # check if start date is greater than end date
    if start_date >= end_date:
        st.error("Start date must be before end date.")

    start_date = str(start_date)
    end_date = str(end_date)

    # Filter the data based on selected date range
    if start_date and end_date:
        filtered_data.index = pd.to_datetime(filtered_data.index)
        filtered_data = filtered_data.loc[start_date:end_date]

    st.header('Economic Behaviors Overtime')
    # Display the filtered data on a line chart
    st.line_chart(filtered_data)
