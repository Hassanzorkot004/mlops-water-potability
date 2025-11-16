import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.neighbors import KNeighborsClassifier

def prepare_data(path):
    #load the dataset
    df = pd.read_csv(path)
    #remove duplicates
    df.drop_duplicates(inplace=True)
     #imputation of missing_values
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    #capping the outliers with IQR method
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "Potability" in numerical_cols:
        numerical_cols.remove("Potability")

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

    #feature engineering

    df["Total_Salts"] = df["Chloramines"] + df["Sulfate"]

    
    #split X/Y
    
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    #Scaling X be
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler

       


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    # ---- TRAIN PREDICTIONS ----
    y_train_pred = model.predict(X_train)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # ---- TEST PREDICTIONS ----
    y_test_pred = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # ---- PRINT RESULTS ----
    print("\n===== TRAIN PERFORMANCE =====")
    print(f"Accuracy : {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall   : {train_recall:.4f}")
    print(f"F1 Score : {train_f1:.4f}")

    print("\n===== TEST PERFORMANCE =====")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    return {
        "train": {"accuracy": train_accuracy, "precision": train_precision,
                  "recall": train_recall, "f1": train_f1},
        "test": {"accuracy": test_accuracy, "precision": test_precision,
                 "recall": test_recall, "f1": test_f1}
    }


def load_model(model_path):
    with open(model_path, "rb") as f:
        modele_charge = pickle.load(f)
    print(f"✅ Modèle chargé depuis : {model_path}")
    
    return modele_charge
    


def save_model(model,model_name):
    with open(model_name+".pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé sous le nom : {model_name}.pkl")




def predict_single(model, input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)
    return int(prediction[0])
    ''' input_dict should be like that:{"ph": 7.3,
    "Hardness": 204.89,
    "Solids": 12000.56,
    "Chloramines": 7.1,
    "Sulfate": 320.45,
    "Conductivity": 492.4,
    "Organic_carbon": 11.72,
    "Trihalomethanes": 65.2,
    "Turbidity": 3.2} dictionnary form'''
