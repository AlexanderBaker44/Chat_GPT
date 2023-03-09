import pandas_datareader as pdr
import pandas as pd
import datetime as dt
import pmdarima as pm
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.model_selection import TimeSeriesSplit
import pickle
from datetime import datetime, timedelta

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


def dickey_fuller(df):
    # Loop through each column of the DataFrame
    for col in df.columns:
        # Run Dickey-Fuller test on the column
        result = adfuller(df[col])
        pvalue = result[1]
        print(f'Column {col}: p-value = {pvalue}')

        # Differentiate the column until it is stationary
        while pvalue > 0.05:
            diff = df[col].diff().dropna()
            result = adfuller(diff)
            pvalue = result[1]
            print(f'  differentiated: p-value = {pvalue}')
            df[col] = diff

    # Filter the DataFrame for rows not older than the given date
    date_string = '2011-01-01'
    df_filtered = df[df.index >= date_string]

    return df_filtered

from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def train_var_model(df_filtered):
    # Split data into training and testing sets
    train_size = int(len(df_filtered) * 0.8)
    train_data, test_data = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]

    # Tune model hyperparameters using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    best_aic = np.inf
    best_order = None
    for p in range(1, 5):
        for q in range(1, 5):
            model = VAR(train_data)
            for train_index, test_index in tscv.split(train_data):
                train = train_data.iloc[train_index]
                test = train_data.iloc[test_index]
                fitted_model = model.fit(maxlags=p, ic='aic', trend='c', method='ols')
                y_hat = fitted_model.forecast(train.to_numpy(), steps=len(test))
                residuals = test.to_numpy() - y_hat
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, q)

    # Train model with best hyperparameters on entire training set
    model = VAR(train_data)
    fitted_model = model.fit(maxlags=best_order[0], ic='aic', trend='c', method='ols')

    # Evaluate model on testing set
    y_hat = fitted_model.forecast(train_data.to_numpy(), steps=len(test_data))
    residuals = test_data.to_numpy() - y_hat
    mse = np.mean(residuals**2)

    # Save model to models folder
    fitted_model.save('models/var_model.pkl')

    return mse


def forecast_var_model(df_filtered):
    # Load the pre-trained VAR model from the models folder
    with open('models/var_model.pkl', 'rb') as f:
        var_model = pickle.load(f)

    # Print the summary of the VAR model
    print(var_model.summary())

    # Forecast
    lag_order = var_model.k_ar
    forecast_input = df_filtered.values[-lag_order:]
    fc = var_model.forecast(y=forecast_input, steps=24)

    # Convert forecast results to dataframe
    fc_df = pd.DataFrame(fc, index=range(df_filtered.shape[0], df_filtered.shape[0]+24), columns=df_filtered.columns + '_forecast')

    # Print forecast results
    print(fc_df)

    # Set start and end dates for the concatenated data
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=730)).strftime("%Y-%m-%d")

    idx = pd.date_range(start=start_date, end=end_date, freq='M')
    fc_df.index = idx
    fc_df.columns = df_filtered.columns
    concatenated_df = pd.concat([fc_df, df_filtered], axis=0)
    concatenated_df.to_csv("data/concatenated_df.csv", index=True)
