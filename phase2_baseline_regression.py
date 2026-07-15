import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def build_regression_model(input_dim):
    # ------------------------------------------------------------
    # Pourquoi PAS de activation='sigmoid' sur la couche de sortie :
    #
    # La sigmoid écrase toute sortie dans l'intervalle (0, 1).
    # Or ici, la cible y (prix médian des maisons en Californie,
    # exprimé en centaines de milliers de dollars) prend des valeurs
    # bien au-delà de 1 (par exemple 2.5, 4.8, jusqu'à 5.0+ dans le
    # dataset California Housing, censuré à 5).
    # ------------------------------------------------------------

    # TODO : instancier un modèle Sequential
    model = keras.Sequential()

    # TODO : ajouter une couche Dense(64, activation='relu', input_shape=(input_dim,))
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))

    # TODO : ajouter une couche Dense(32, activation='relu')
    model.add(layers.Dense(32, activation='relu'))

    # TODO : ajouter la couche de sortie : Dense(1) sans activation (régression = valeur continue)
    model.add(layers.Dense(1))

    # TODO : compiler avec optimizer='adam', loss='mse', metrics=['mae']
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # TODO : retourner le modèle
    return model


# ------------------------------------------------------------
# Préparation des données (reprise de la Phase 1)
# ------------------------------------------------------------
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# ------------------------------------------------------------
# Construction et résumé du modèle
# ------------------------------------------------------------
model = build_regression_model(input_dim=8)
model.summary()

# TODO : appeler model.fit avec X_train_norm, y_train, epochs=100, batch_size=32,
# validation_data=(X_val_norm, y_val), verbose=1
# stocker le retour dans `history`
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_norm, y_val),
    verbose=1
)

# TODO : appeler model.evaluate sur (X_test_norm, y_test, verbose=0)
# récupérer (test_loss, test_mae)
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)

print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")