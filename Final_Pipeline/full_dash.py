from multi_select_dash import display_data
from var_display import display_selected_value
from utils.update_functions import get_fred_data, fit_and_save_models
from utils.update_functions import train_var_model, forecast_var_model, dickey_fuller
import streamlit as st
import datetime
import calendar

# Get today's date
today = datetime.date.today()

# Check if it's the end of the month
if today.day == calendar.monthrange(today.year, today.month)[1]:
    # Get FRED data
    data, _ = get_fred_data()
    if _:
        print("The condition is true.")
        # Fit and save models
        fit_and_save_models(data)
        df_filtered = dicky_fuller(df)
        train_var_model(df_filtered)
        forecast_var_model(df_filtered)

# Define your Streamlit app pages
PAGES = {
    "Display Data": display_data,
    "Display Selected Value": display_selected_value
}

# Define a function to create the page navigation
def page_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select your page", list(PAGES.keys()))
    PAGES[page]()  # Call the selected page function

# Run the Streamlit app
if __name__ == "__main__":
    page_navigation()
