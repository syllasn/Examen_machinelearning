# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pickle


# Function to load data (5 pts)
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data
    

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    data['date'] = pd.to_numeric(data['date'])
    return data

def encode_categorical(data):
    label_encoder = LabelEncoder()
    data['Facility'] = label_encoder.fit_transform(data['Facility'])
    return data

def handle_missing_values(data):
    data['Value'].fillna(data['Value'].mean(), inplace=True)
    return data

# Function to split data into training and testing sets (5 pts)
def split_data(data, test_size=0.2, random_state=42): 
    # Split data into training (80%) and testing (20%) sets
    X = data[['date', 'Facility']]
    y = data['Value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    return X_train, X_test, y_train, y_test

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse



# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, model_file):
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

# Main function
def main():
    # Load data
    file_path = "Malaria.xlsx"
    data = load_data(file_path)
    #print(data.head())
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data.head())

    # Encode categorical variables
    df = encode_categorical(preprocessed_data)
    

    df = handle_missing_values(df)
    print(df.head())
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    print(best_model)
    
    # Evaluate the model
    mse = evaluate_model(best_model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    
    # Deploy the model (bonus)
    with open('malaria_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

if __name__ == "__main__":
    main()
