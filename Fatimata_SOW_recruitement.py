# import libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load data (5 pts)
def load_data(file_path):
    try:
        # Charger les données à partir du fichier spécifié
        data = pd.read_excel(file_path)
        return data

    except Exception as e:
        print(f"Une erreur s'est produite lors du chargement des données : {str(e)}")
        return None


    data = load_data(r'C:\Users\FATIMATA SOW\Examen_machinelearning\malaria.xlsx')


# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Gérer les valeurs manquantes en utilisant l'imputation moyenne
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)

    # Standardiser les données
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data

# Function to split data into training and testing sets (5 pts)
def split_data(data):
    X = data.drop('target_column', axis=1)  # Remplacez 'target_column' par le nom de votre colonne cible
    y = data['target_column']  # Remplacez 'target_column' par le nom de votre colonne cible

    # Diviser les données en ensembles d'entraînement (80%) et de test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train):
    # Définir les hyperparamètres à ajuster
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        # Ajoutez d'autres hyperparamètres à ajuster ici
    }

    # Créer un modèle (par exemple, RandomForestClassifier)
    model = RandomForestClassifier()

    # Effectuer une recherche par grille pour ajuster les hyperparamètres
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Sélectionner le meilleur modèle
    best_model = grid_search.best_estimator_

    return best_model
# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Prédire les étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer les métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Afficher les métriques
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

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
