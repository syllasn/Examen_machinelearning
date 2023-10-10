# import libraries
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    data = pd.read_excel(file_path, sheet_name='Feuille 1')
    return data

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    # Deal with outlier data 
    # return data
    
    # Drop any rows with missing values
    data.dropna(inplace=True)

    # Convert the 'date' column to a datetime object
    data['date'] = pd.to_datetime(data['date'])

    # define 'date' as index
    data.set_index('date', inplace=True)
    return data

# Function to split data into training and testing sets (5 pts)
def split_data(preprocessed_data): 
    # Split data into training (80%) and testing (20%) sets
    train, test = train_test_split(preprocessed_data, test_size=0.2, random_state=42)
    return train, test

# Function to train a model with hyperparameters (30 pts)
def train_model(data): 
    best_model = None
    best_aic = np.inf
    best_model_type = None

    # MA Model
    for q in range(1, 6):
        model = ARIMA(data, order=(0, 0, q))
        try:
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_model = results
                best_model_type = "MA"
        except:
            continue

    # ARIMA Model
    for p in range(1, 6):
        for d in range(1, 3):
            for q in range(1, 6):
                model = ARIMA(data, order=(p, d, q))
                try:
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_model = results
                        best_model_type = "ARIMA"
                except:
                    continue

    # SARIMAX Model
    for p in range(1, 6):
        for d in range(1, 3):
            for q in range(1, 6):
                for P in range(1, 4):
                    for D in range(1, 3):
                        for Q in range(1, 4):
                            model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, 12))
                            try:
                                results = model.fit()
                                aic = results.aic
                                if aic < best_aic:
                                    best_aic = aic
                                    best_model = results
                                    best_model_type = "SARIMAX"
                            except:
                                continue

    return best_model, best_model_type



# Function to evaluate the model (15 pts)
def evaluate_model(model, data):
    # Evaluate the best model 
    predictions = model.get_forecast(steps=len(data)).predicted_mean

    # Calculer le RMSE et le MAE
    rmse = np.sqrt(mean_squared_error(data, predictions))
    mae = mean_absolute_error(data, predictions)

    return rmse, mae

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, test):
    # Deploy the best model using Streamlit or Flask (bonus)
    pass

# Main function
def main():
    # Load data
    data = load_data("malaria.xlsx")
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data
    train, test = split_data(preprocessed_data)
    
    # Train a model with hyperparameters
    best_model = train_model(train)
    
    # Evaluate the model
    evaluate_model(best_model, test)
    
    # Deploy the model (bonus)
    deploy_model(best_model, test)

if __name__ == "__main__":
    main()
