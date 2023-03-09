import streamlit as st
import pandas as pd
import pickle

def display_selected_value():
    # Load dataset from data folder
    df = pd.read_csv("data/concatenated_df.csv", index_col=0)

    # Create a list of column names
    cols = list(df.columns)

    # Add a widget to select the first value
    value1 = st.selectbox('Select a value from the list', cols)

    # Remove the first value from the list
    #cols.remove(value1)

    cols = list(df.columns)
    # Add a widget to select the second value
    value2 = st.selectbox('Select another value from the list', cols)

    # Display the selected values
    st.write('Selected values:', value1, 'and', value2)

    # Load the pre-trained VAR model from the models folder
    with open('models/var_model.pkl', 'rb') as f:
        var_model = pickle.load(f)

    row_index = value1
    col_index = value2

    # assuming `row_index` and `col_index` are the selected row and column indices from the selectbox
    cov_param = var_model.cov_ybar()
    cov_param_df = pd.DataFrame(cov_param, columns=df.columns, index=df.columns)
    # set the selected row and column from cov_param_df to a variable
    selected_value = cov_param_df.loc[row_index, col_index]
    # display the selected value on the dash
    rounded_value = round(selected_value, 5)
    st.markdown(f"<h1>{rounded_value}</h1>", unsafe_allow_html=True)
