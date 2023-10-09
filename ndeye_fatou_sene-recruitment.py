# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    df = pd.read_excel(file_path)
    return df
    

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    # This method help to drop NaN values
    data_nn = data.dropna()
    # Deal with outlier data 
    num_col = ["Value"]
    for i in num_col:
        col_skew = data_nn[i].skew()
        print(f"The skew for column {i} is {col_skew}")
        if (col_skew < -1) | (col_skew > 1):
            print(f"Column {i} has outliers")
            data_nn[i] = np.where(data_nn[i] < data_nn[i].quantile(0.1), data_nn[i].quantile(0.1), data_nn[i])
            data_nn[i] = np.where(data_nn[i] > data_nn[i].quantile(0.9), data_nn[i].quantile(0.9), data_nn[i])
            print(f" The column {i} has been processed and the skew for column {i} is now {data_nn[i].skew()}")
        else:
            print(f"There are not outliers at column {i}")
    
    # return data
    return data_nn
    

# Function to split data into training and testing sets (5 pts)
def split_data(data): 
    # Split data into training (80%) and testing (20%) sets
    y = data["Value"]
    X = data["date"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test
    

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    # Train a or many models with hyperparameter tuning
    # Return best model
    pass

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
    data = load_data("malaria.xlsx")
    
     # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    print(X_train)
    
    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
    
    # Deploy the model (bonus)
    deploy_model(best_model, X_test)

if __name__ == "__main__":
    main()
