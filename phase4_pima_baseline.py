import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement Pima (URL directe, pas de compte Kaggle requis)
# Attention : l'URL doit contenir un tiret entre "indians" et "diabetes",
# sinon 404 (pima-indiansdiabetes.data.csv n'existe pas).
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)

# ------------------------------------------------------------
# TODO : afficher df['Outcome'].value_counts() pour voir la distribution de classes
# noter les proportions exactes avant d'entraîner quoi que ce soit
# ------------------------------------------------------------
print("Répartition des classes (Outcome) :")
print(df['Outcome'].value_counts())
print("np.bincount(y) :", np.bincount(df['Outcome'].values))

# ------------------------------------------------------------
# RÉPARTITION EXACTE (768 exemples au total) :
#   Classe 0 (pas de diabète) : 500 exemples -> 500/768 = 65.10 %
#   Classe 1 (diabète)        : 268 exemples -> 268/768 = 34.90 %
#
# QUESTION POSÉE AVANT LE TRAINING : un modèle qui prédirait TOUJOURS
# la classe majoritaire (0) afficherait quelle accuracy ?
#   -> RÉPONSE : 500/768 ≈ 0.6510, soit environ 65 % d'accuracy.
#
# Cette baseline "bête" à 65 % est le seuil minimum de comparaison.
# Un modèle entraîné qui n'atteint pas significativement plus que 65 %
# n'a rien appris d'utile — il a juste appris à dire "pas de diabète"
# tout le temps, ce que l'accuracy seule ne permet PAS de distinguer
# d'un vrai apprentissage (d'où l'intérêt de vérifier aussi
# model.predict(X_val).mean(), qui doit être proche de 0.35 et pas
# de 0.05 : si le modèle prédit presque toujours des probabilités
# proches de 0, c'est le signe qu'il s'est effondré sur la classe
# majoritaire plutôt que d'avoir appris à distinguer les deux classes).
# ------------------------------------------------------------

# ------------------------------------------------------------
# TODO : afficher (df == 0).sum() pour toutes les colonnes
# Glucose=0, BMI=0, Insulin=0, SkinThickness=0 sont physiologiquement impossibles
# ce sont des NaN déguisés en zéros, encodage courant dans les datasets médicaux anciens
# on les laisse pour l'instant, mais les noter : c'est un point de fragilité réel
# ------------------------------------------------------------
print("\nNombre de zéros par colonne :")
print((df == 0).sum())
# Note : Insulin (374 zéros, ~49% !), SkinThickness (227, ~30%),
# BloodPressure (35), BMI (11), Glucose (5) contiennent des zéros
# médicalement impossibles = valeurs manquantes non traitées.
# Pregnancies=0 et Outcome=0 sont légitimes (0 grossesse, classe
# négative) donc pas concernés par ce problème.

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# TODO : split train/test 80/20, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO : instancier StandardScaler, le fitter sur X_train UNIQUEMENT, transformer train et test
# (fitter sur X_test = data leakage : le modèle verrait les stats du test avant évaluation)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

# TODO : construire un modèle Sequential binaire
# architecture : Dense(64, relu, input_shape=(8,)) -> Dense(32, relu) -> Dense(1, sigmoid)
# compiler avec optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
model = keras.Sequential()
model.add(keras.Input(shape=(8,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TODO : entraîner 100 epochs, validation_split=0.2, batch_size=32
# stocker le résultat dans une variable `history`
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=0
)

# TODO : afficher la val_accuracy finale (max sur toutes les epochs)
# et vérifier que model.predict(X_val).mean() est proche de 0.35 (pas 0.05)
best_val_accuracy = max(history.history['val_accuracy'])
print(f"\nVal accuracy max sur les 100 epochs : {best_val_accuracy:.4f}")
print(f"(à comparer à la baseline 'classe majoritaire' : {500/768:.4f})")

# Reconstruire un split de validation identique à celui utilisé en interne
# par validation_split=0.2 n'est pas trivial (Keras prend les X% derniers
# exemples de X_train_norm) -> on le refait explicitement pour pouvoir
# calculer predict(X_val).mean() proprement.
n_val = int(len(X_train_norm) * 0.2)
X_val_check = X_train_norm[-n_val:]
val_predictions = model.predict(X_val_check, verbose=0)
print(f"model.predict(X_val).mean() = {val_predictions.mean():.4f}")
print("(doit être proche de 0.35 = proportion réelle de la classe 1 dans X_val ;")
print(" proche de 0.05 indiquerait un modèle effondré sur la classe majoritaire)")