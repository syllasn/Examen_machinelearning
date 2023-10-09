# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    data = pd.read_excel(file_path)
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

    # Encode categorical variables (Facility) using one-hot encoding
    data = pd.get_dummies(data, columns=['Facility'], drop_first=True)

    return data

# Function to split data into training and testing sets (5 pts)
def split_data(preprocessed_data): 
    # Split data into training (80%) and testing (20%) sets
    # Define your features (X) and target variable (y)
    X = preprocessed_data.drop(columns=['Value'])  # Features (all columns except 'Value')
    y = preprocessed_data['Value']  # Target variable

    # Split the data into training and test sets (e.g., 80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
