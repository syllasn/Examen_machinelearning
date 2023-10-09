import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse
# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    return pd.read_excel(file_path)
    
# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    data['Value'].fillna(data['Value'].mean(), inplace=True)
    # Deal with outlier data 
    Q3 = np.percentile(data['Value'], 90, method='midpoint')
    data["Value"] = np.where(data["Value"] >Q3, Q3,data['Value'])
    return data

# Function to split data into training and testing sets (5 pts)
def split_data(data): 
    # Split data into training (80%) and testing (20%) sets
    labelencoder = LabelEncoder()
    X_ = labelencoder.fit_transform(data['date'])
    X = labelencoder.fit_transform(data['Facility'])
    y = data["Value"]
    X = pd.DataFrame({'date' : X_,
                      'Facility' : X},
                        columns=['date','Facility'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test

def gridsearch(models, params,train_x, train_y):
  model_best=[]
  accuracy_best=[]
  for i in range(len(models)):
    print(f'tour numero: {i}')
    model=GridSearchCV(models[i], params[i], cv=5)
    model.fit(train_x, train_y)
    print(model.best_estimator_)
    print(model.best_score_)
    model_best.append(model.best_estimator_)
    accuracy_best.append(model.best_score_)
  return model_best, accuracy_best

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    param_grid_linr = [
        {'fit_intercept':[True,False]},
        {'copy_X':[True,False]}, 
        {'n_jobs':[True,False]}, 
        {'positive':[True,False]}
    ]
    param_grid_logis = [{
        {'kernel':["linear", "poly", "rbf", "sigmoid", "precomputed"]},
        {'gamma':["scale", "auto"]},
        {'shrinking':[True,False]}
    }]
    param_grid_ridge= [{
        {'copy_X':[True,False]},
        {'fit_intercept':[True,False]},
        {"alpha": [1.0, 1.5, 0.5, 2.0] }
    }]
    models = []
    models.append(LinearRegression())
    models.append(Ridge())
    models.append(SVR())

    parametres=[]
    parametres.append(param_grid_linr)
    parametres.append(param_grid_ridge)
    parametres.append(param_grid_logis)

    mod, accu = gridsearch(models, parametres, X_train, y_train)
    for i in range(len(mod)):
        if accu[i]== max(accu):
            with open("model.pkl", 'wb') as model_file:
                pkl.dump(mod[i], model_file) 
            return mod[i]

# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Evaluate the best model 
    y_pred = model.predict(X_test)
    
    return mse(y_test, y_pred)

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    # Deploy the best model using Streamlit or Flask (bonus)
    pass

# Main function
def main():
    # Load data
    data = load_data("./malaria.xlsx")
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    
    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
    
    # Deploy the model (bonus)
    """deploy_model(best_model, X_test)"""

if __name__ == "__main__":
    main()
