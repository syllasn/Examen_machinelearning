# import libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import streamlit as st
import plotly.express as px


# Function to load data (5 pts)
def load_data(file_path, sheet_name):
    # Load data from the CSV file or another format and return data
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    return df
    

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    k = 2
    imputer = KNNImputer(n_neighbors=k)
    data["Value"] = imputer.fit_transform(data)
    # Deal with outlier data 
    lower_bound = data['Value'].quantile(0.05)
    upper_bound = data['Value'].quantile(0.95)

    data['Value'] = np.where(data['Value'] < lower_bound, lower_bound, data['Value'])
    data['Value'] = np.where(data['Value'] > upper_bound, upper_bound, data['Value'])

    # return data
    return data

# Function to split data into training and testing sets (5 pts)
def split_data(data): 
    # Split data into training (80%) and testing (20%) sets
    len_train = int(len(data) * 0.8)
    train = data[:len_train]
    test = data[len_train:]

    return train, test

# Function to train a model with hyperparameters (30 pts)
def train_model(train): 
    # Train a or many models with hyperparameter tuning
    prophet_model = Prophet()
    prophet_data_train = train.reset_index()
    prophet_data_train.rename(columns={"date":"ds", "Value":"y"}, inplace=True)
    prophet_model.fit(prophet_data_train)
    # Return best model
    return prophet_model

# Function to evaluate the model (15 pts)
def evaluate_model(model, test):
    # Evaluate the best model 
    to_predict = model.make_future_dataframe(periods=len(test))
    prophet_predictions = model.predict(to_predict)["yhat"][-len(test):]
    mae = mean_absolute_error(test["Value"], prophet_predictions)
    print(f"Mean absolute error: {mae}")
    

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, test):
    # Deploy the best model using Streamlit or Flask (bonus)
    st.title('Time Series Forecasting with Prophet')
    num_days = st.slider("Number of days for future prediction:", 1, len(test))
    to_predict = model.make_future_dataframe(periods=num_days)

    forecast = model.predict(to_predict)["yhat"][-len(test):]
    test = test.reset_index()
    st.subheader("Test Data")
    st.write(test)
    st.subheader("Test Data Plot")
    fig_test_data = px.line(test, x='date', y='Value', title='Test Data')
    st.plotly_chart(fig_test_data)

    st.subheader("Forecasted Data")
    fig_forecast = px.line(forecast, y='yhat', title='Forecasted Data')
    st.plotly_chart(fig_forecast)


# Main function
def main():
    # Load data
    data = load_data("malaria.xlsx", "Feuille 1")
    
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
