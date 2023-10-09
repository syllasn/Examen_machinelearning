# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


# Function to load data (5 pts)
def load_data(file_path = 'malaria.xlsx'):
    # Load data from the CSV file or another format and return data
    data = pd.read_excel(file_path)
    return data
    

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    data['Value'] = data['Value'].fillna(data['Value'].mean())
    # Deal with outlier data
    data['Value'] = np.where(data['Value'] < data['Value'].quantile(0.10), data['Value'].quantile(0.10), data['Value'])
    # return data
    return data

# Function to split data into training and testing sets (5 pts)
def split_data(data): 
    # Split data into training (80%) and testing (20%) sets
    y = data['Value']
    X = data['date']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    # Train a or many models with hyperparameter tuning
    # Return best model

    # Specify the ARIMA order (p, d, q)
    p = 1  # Replace with your chosen p value
    d = 1  # Replace with your chosen d value
    q = 1  # Replace with your chosen q value

    # Fit an ARIMA model to each time series in your data
    # You can loop through your data if you have multiple time series
    model = sm.tsa.ARIMA(y_train, order=(p, d, q))
    results = model.fit()

    # Make predictions
    forecast_steps = 10
    forecast, stderr, conf_int = results.forecast(steps=forecast_steps)

    print(forecast)


    return model

# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Evaluate the best model

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    # Deploy the best model using Streamlit or Flask (bonus)


    pass

# Main function
def main():
    # Load data
    data = load_data('malaria.xlsx')

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)

    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)

    print()
    
    # Deploy the model (bonus)
    deploy_model(best_model, X_test)

if __name__ == "__main__":

    main()
