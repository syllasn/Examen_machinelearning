# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
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
def train_model(X_train, y_train): 
    # Define a grid of hyperparameters to search through
    param_grid = {
        'p': range(0, 3),  # AR order
        'd': range(0, 2),  # I order
        'q': range(0, 3),  # MA order
        'P': range(0, 3),  # Seasonal AR order
        'D': range(0, 2),  # Seasonal I order
        'Q': range(0, 3),  # Seasonal MA order
        's': [12],         # Seasonal period (assuming monthly data)
    }

    best_model = None
    best_mse = float('inf')

    # Iterate through all possible combinations of hyperparameters
    for params in ParameterGrid(param_grid):
        try:
            # Fit a SARIMA model with the current hyperparameters
            model = SARIMAX(y_train, exog=X_train, order=(params['p'], params['d'], params['q']),
                            seasonal_order=(params['P'], params['D'], params['Q'], params['s']))
            results = model.fit()

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_train, results.fittedvalues)

            # Update the best model if this one has a lower MSE
            if mse < best_mse:
                best_mse = mse
                best_model = results
        except:
            continue

    # The best model is not returned, it's available for use outside of this function
    return


# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Evaluate the best model 
    pass

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    # Deploy the best model using Streamlit or Flask (bonus)
    pass

# Main function
def main():
    # Load data
    data = load_data("link")
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    
    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
    
    # Deploy the model (bonus)
    deploy_model(best_model, X_test)

if __name__ == "__main__":
    main()
