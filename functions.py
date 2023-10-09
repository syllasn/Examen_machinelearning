# import libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st

import warnings

# Silence all warnings to avoid unecessary clutter
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the Excel file
    data = pd.read_excel(file_path, sheet_name="Feuille 1")
    return data


# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    # Deal with outlier data
    # return data
    data['Value'].fillna(data['Value'].mean(), inplace=True)
    z_scores = (data['Value'] - data['Value'].mean()) / data['Value'].std()
    data['Value'] = z_scores.abs() < 3
    data = data[data['Value']]
    return data


# Function to split data into training and testing sets (5 pts)
def split_data(data):
    # Split data into features (X) and target variable (y)
    X = data.index.to_frame()
    y = data['Value']

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train):
    # We train an ARIMA model
    model = ARIMA(y_train, order=(5, 1, 0)) 
    model_fit = model.fit()

    return model_fit



# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Get the last date in the training set
    last_date = X_test.index[-1]

    # Create a forecast index for the next N periods
    forecast_index = pd.date_range(start=last_date, periods=len(X_test) + 1, freq='M')[1:]

    # Make predictions on the test set
    forecast = model.get_forecast(steps=len(X_test), index=forecast_index)

    # Get the predicted values
    y_pred = forecast.predicted_mean

    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

# Function to deploy the model using Streamlit (bonus) (10 pts)
def deploy_model(model, X_test):
    # Streamlit app code
    st.title('Malaria Cases Prediction App')

    # Set default date range to 2018-04-01 to 2023-03-01
    default_start_date = pd.to_datetime('2018-04-01')
    default_end_date = pd.to_datetime('2023-03-01')

    # Input date range for prediction
    date_range = st.date_input('Select Date Range', [default_start_date, default_end_date], key='date_range')

    # Ensure date_range is in the correct format
    start_date, end_date = pd.to_datetime(date_range)

    # Make predictions using the deployed model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        forecast = model.get_forecast(steps=(end_date - start_date).days + 1, index=pd.date_range(start=start_date, end=end_date, freq='D'))

    # Extract the predicted mean
    predicted_mean = forecast.predicted_mean
    
    # Convert datetime index to strings for plotting
    predicted_mean.index = predicted_mean.index.strftime('%Y-%m-%d')

    # Display predictions
    st.subheader('Predicted Mean')
    st.line_chart(predicted_mean)

    # Print the summary of the ARIMA model
    st.subheader('ARIMA Model Summary')
    st.text(str(model.summary()))

# Main function
def main():
    # Load data
    data = load_data("malaria.xlsx")

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)

    # Train an ARIMA model
    best_model = train_model(X_train, y_train)

    # Evaluate the ARIMA model
    evaluate_model(best_model, X_test, y_test)
    
    # Deploy the model (bonus)
    deploy_model(best_model, X_test)


if __name__ == "__main__":
    main()
