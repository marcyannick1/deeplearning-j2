# ============================================================
# CHOIX DU PIPELINE : split PUIS scaler.fit(X_train) — option (b)
# ============================================================
# On NE fait PAS scaler.fit(X) sur tout le dataset avant de split.

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger le dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# TODO : faire un premier split train/test avec test_size=0.2 et random_state=42
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO : faire un second split train/val sur le résultat précédent (val_size=0.2 du train)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# TODO : instancier un StandardScaler et le fitter sur X_train UNIQUEMENT
scaler = StandardScaler()
scaler.fit(X_train)

# TODO : transformer X_train, X_val, X_test avec le scaler fitté
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# TODO : afficher les shapes de X_train, X_val, X_test
print("Shapes :")
print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

# TODO : afficher les stats descriptives de X_train_norm (mean et std par feature)
print("\nStats descriptives de X_train_norm :")
print("Mean par feature :", X_train_norm.mean(axis=0))
print("Std par feature  :", X_train_norm.std(axis=0))

# TODO : afficher les feature_names du dataset ET vérifier qu'il y en a bien 8
print("\nFeature names :", housing.feature_names)
print("Nombre de features :", len(housing.feature_names))
assert len(housing.feature_names) == 8, "Il devrait y avoir 8 features"