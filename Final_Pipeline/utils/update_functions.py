import pandas_datareader as pdr
import pandas as pd
import datetime as dt
import pmdarima as pm
import os

def get_fred_data():
    # Define series to pull
    series = ['GDPC1', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'CIVPART', 'INDPRO',
              'TOTALSA', 'DSPIC96', 'PCE', 'DGS10', 'CPALTT01USM657N']

    # Set end date as current date
    end_date = dt.datetime.now().date()

    # Set start date as January 1st, 2013
    start_date = pd.to_datetime('2013-01-01').date()

    # Pull data from FRED
    data = pdr.get_data_fred(series, start_date, end_date)

    # Resample to monthly frequency
    data = data.resample('M').mean()

    # Check if the final row contains nulls
    final_row = data.iloc[-1]
    if final_row.isnull().any():
        result = False
    else:
        result = True

    # Print first and last 5 rows of data
    print(data.head())
    print(data.tail())

    return data, result



def fit_and_save_models(data):
    """
    Fits an auto_arima model to each column in the input dataset and saves
    the models to the "models" folder in the current working directory.

    Args:
    data (pandas.DataFrame): The input dataset to fit models to.

    Returns:
    None
    """

    # Create the "models" folder if it doesn't already exist
    if not os.path.exists("models"):
        os.mkdir("models")

    # Loop over each column in the input dataset
    for col in data.columns:

        # Fit an auto_arima model to the current column
        model = pm.auto_arima(data[col], seasonal=True, m=12)

        # Save the model to a pickle file in the "models" folder
        model_filename = f"models/{col}.pickle"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
    # Load the saved models
    models = {}
    for col in fred_data_scaled.columns:
        with open(f"models/{col}.pickle", "rb") as f:
            models[col] = pickle.load(f)

    # Make predictions
    preds = pd.DataFrame()
    for col in fred_data_scaled.columns:
        model = models[col]
        pred, _ = model.predict(n_periods=24, return_conf_int=True)
        preds[col] = pred

    # Save the predictions to a non-indexed CSV file
    preds.to_csv("data/predictions.csv", index=False)
    # Concatenate preds and fred_data_scaled
    concatenated_df = pd.concat([fred_scaled_data, preds], ignore_index=False)

    # Save concatenated dataframe to CSV
    concat_data.to_csv('data/fred_data_scaled_with_preds.csv', index=True)
