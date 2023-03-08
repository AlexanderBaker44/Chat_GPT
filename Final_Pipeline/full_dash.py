from multi_select_dash import display_data
from utils.update_functions import get_fred_data, fit_and_save_models
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

display_data()
