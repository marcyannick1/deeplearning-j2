import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def build_regression_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=100):
    """Entraîne un modèle de régression avec un callback TensorBoard horodaté."""

    # TODO : construire log_dir = f"logs/fit/{run_name}_" + timestamp au format HHMMSS
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_{timestamp}"

    # TODO : instancier keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # TODO : instancier le modèle via build_regression_model(input_dim=8)
    model = build_regression_model(input_dim=8)

    # TODO : appeler model.fit avec callbacks=[tb_callback],
    # validation_data=(X_val, y_val), epochs=epochs, batch_size=32, verbose=0
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0
    )

    # TODO : afficher un message confirmant le run et son chemin de logs
    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")

    # TODO : retourner (model, history)
    return model, history


# ------------------------------------------------------------
# Préparation des données (reprise des Phases 1 et 2)
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

# Run 1 : données normalisées (bon comportement attendu)
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val, run_name="california_norm"
)

# Run 2 : données brutes (comportement dégradé à observer)
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val, run_name="california_raw"
)

# Lancer TensorBoard dans un terminal séparé :
# tensorboard --logdir=logs/fit
# Puis ouvrir http://localhost:6006

# ------------------------------------------------------------
# HYPOTHÈSE / INTERPRÉTATION ATTENDUE (à VÉRIFIER dans TensorBoard,
# scalars train_loss / val_loss — je n'ai pas pu exécuter ce script
# de mon côté, le téléchargement du dataset a été bloqué par le
# réseau restreint de mon environnement d'exécution)
# ------------------------------------------------------------
# Run "california_norm" (features standardisées) :
#   -> Situation (a) attendue. train_loss et val_loss devraient
#   descendre ensemble, de façon régulière, vers un MAE de test
#   raisonnable (~0.4-0.5, soit 40-50k$ d'erreur moyenne). Le scaler
#   a été fitté sur X_train uniquement (Phase 1, pas de fuite), et
#   les features à échelle comparable permettent au réseau de
#   converger vite et stablement avec Adam.
#
# Run "california_raw" (features NON normalisées) :
#   -> Comportement dégradé attendu : convergence plus lente et
#   plus instable epoch par epoch (loss qui oscille, parfois des
#   pics), car les features ont des échelles très différentes
#   (ex. MedInc ~0-15 vs Population ~3-35000). Le gradient est
#   dominé par les features à grande échelle, ce qui perturbe
#   l'optimisation. Ça ne correspond pas forcément à de l'overfitting
#   classique (b) : le problème est l'optimisation elle-même
#   (features non mises à l'échelle), pas la variance du modèle.
#   -> À CONFIRMER en regardant les deux courbes côte à côte dans
#   TensorBoard : si train et val du run "raw" divergent nettement
#   l'une de l'autre en plus d'être instables, ce serait plutôt (b).
#
# Situation (c) non attendue ici : val_loss ne devrait pas passer
# sous train_loss, puisque le split a été fait AVANT le fit du
# scaler (pas de fuite, cf. Phase 1). Si (c) apparaît quand même à
# l'exécution, le premier réflexe est de revérifier que le scaler
# n'a pas été fitté sur l'ensemble complet (X au lieu de X_train).