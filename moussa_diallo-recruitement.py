# import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from flask import Flask, render_template, request
import joblib
# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    data = pd.read_excel(file_path, parse_dates=True)
    return data
    

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    data["Year"] = data["date"].dt.year
    data["Month"] = data["date"].dt.month
    # Handle missing data using appropriate imputation
    print(data.shape)
    print(data.isna().sum())
    mean_value = data["Value"].mean()
    data["Value"].fillna(mean_value, inplace = True)
    print(data.isna().sum())
    # Deal with outlier data
    print("before remove outlier", data.shape)
    lower = data['Value'].quantile(0.05)
    upper = data['Value'].quantile(0.95)
    data = data[(data['Value'] >= lower) & (data['Value'] <= upper)]
    print("after remove outlier", data.shape)
    data.index = pd.DatetimeIndex(data.index, freq = None) 
    data.sort_index(inplace=True)
    return data

# Function to split data into training and testing sets (5 pts)
def split_data(data): 
    # Split data into training (80%) and testing (20%) sets
    X = data[["Facility","Month","Year"]]
    X = pd.get_dummies(X, columns=['Facility'], prefix='Facility')
    #X = X
    print(X.columns)
    y = data.Value
    #data['Value_Lag1'] = data['Value'].shift(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    # Train a or many models with hyperparameter tuning
    # Return best model
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    #arima_model = ARIMA(y_train, order=(5,1,0))

    linear_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    #arima_model_fit = arima_model.fit()
    #model.fit(X_train, y_train)
    # Choose the best model based on RMSE
    # Make predictions with each model on the test data
    linear_predictions = linear_model.predict(X_train)
    rf_predictions = rf_model.predict(X_train)
    #arima_predictions = arima_model_fit.forecast(steps=len(X_train))
    # Evaluate the models using Root Mean Squared Error (RMSE)
    linear_rmse = np.sqrt(mean_squared_error(y_train, linear_predictions))
    rf_rmse = np.sqrt(mean_squared_error(y_train, rf_predictions))
    #arima_rmse = np.sqrt(mean_squared_error(y_test, arima_predictions))

    # Print RMSE for each model
    print(f"Linear Regression RMSE: {linear_rmse:.2f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    #print(f"ARIMA RMSE: {arima_rmse:.2f}")
    best_model = None
    if linear_rmse < rf_rmse :# and linear_rmse < arima_rmse:
        best_model = linear_model
        print("Linear Regression is the best model.")
    elif rf_rmse < linear_rmse and rf_rmse : #< arima_rmse:
        best_model = rf_model
        print("Random Forest is the best model.")
    """
        else:
        best_model = arima_model_fit
        print("ARIMA is the best model.")
    """
    return best_model
# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Evaluate the best model 
    # Evaluate the best model using appropriate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    return rmse

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    # Deploy the best model using Streamlit or Flask (bonus)

    app = Flask(__name__)

    # Load your trained model (replace with your actual model loading code)
    
    #model = joblib.load('your_model.pkl')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        # Get user input from the web form
        # Modify this part to match your input features
        facility = request.form.get('Facility')
        year = request.form.get('Year')
        month = request.form.get('Month')
        #value = float(request.form.get('Value'))

        # Perform any necessary preprocessing on the input data
        # Create a feature vector for prediction
        feature_vector = [facility,year,month]  # Modify this based on your input features

        # Make a prediction using your model
        prediction = model.predict([feature_vector])

        return render_template('result.html', prediction=prediction[0])

    if __name__ == '__main__':
        app.run(debug=True)


# Main function
def main():
    # Load data
    data = load_data("malaria.xlsx")
    print(data.head())
    
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
