import argparse
from sklearn.neighbors import KNeighborsClassifier
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

# Chemin du dataset
PATH = "/home/hassan_004/Hassan-Zorkot-4DS8-ml_project/water_potability.csv"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()

    # Étape 1 : Préparation des données
    if args.prepare or args.train or args.validate:
        print("📌 Préparation des données...")
        X_train, X_test, y_train, y_test, scaler = prepare_data(PATH)

    # Étape 2 : Entraînement
    if args.train:
        print("📌 Entraînement du modèle KNN...")
        model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        model = train_model(model, X_train, y_train)
        save_model(model, "model_KNN")

    # Étape 3 : Évaluation
    if args.validate:
        print("📌 Chargement et validation du modèle...")
        model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        model = train_model(model, X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()




