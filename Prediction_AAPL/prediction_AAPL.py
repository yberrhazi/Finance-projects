import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


ticker = "AAPL"
stock = yf.Ticker(ticker)
historical_data = stock.history(period="5y", interval = "1d")  # data for the last five years for more accuracy
print("Historical Data:")


historical_data["return"] = historical_data["Close"].pct_change() #pct_change s'occupe de calculer le pourcentage de changement d'un jour à l'autre, return = (close (j) -close(j-1))/ close(j-1)
#print(historical_data)

# 1) Créer une colonne qui contient le prix de "demain"
historical_data["Close_tomorrow"] = historical_data["Close"].shift(-1)
# 2) Comparer le prix de demain avec celui d'aujourd'hui
historical_data["target"] = (historical_data["Close_tomorrow"] > historical_data["Close"]).astype(int)
# 3) Enlever la dernière ligne (elle a un NaN dans Close_tomorrow)
historical_data = historical_data.dropna()
#print(historical_data[["Close", "return", "Close_tomorrow", "target"]].head())

# Moyenne mobile courte (5 jours) et longue (10 jours)
historical_data["sma_5"] = historical_data["Close"].rolling(window=5).mean()
historical_data["sma_10"] = historical_data["Close"].rolling(window=10).mean()

# Différence normalisée entre les deux
historical_data["sma_diff"] = (historical_data["sma_5"] - historical_data["sma_10"]) / historical_data["Close"]

# Supprimer toutes les lignes avec au moins un NaN
historical_data = historical_data.dropna()
print(historical_data[["Close", "return", "sma_5", "sma_10", "sma_diff", "target"]].head())

feature_cols = ["return", "sma_diff"]
X = historical_data[feature_cols]
y = historical_data["target"]
# On réserve les 200 derniers jours pour le test
test_size = 200

X_train = X.iloc[:-test_size]
y_train = y.iloc[:-test_size]

X_test = X.iloc[-test_size:]
y_test = y.iloc[-test_size:]
print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Proportion de 1 dans target (train):", y_train.mean())
print("Proportion de 1 dans target (test):", y_test.mean())


# Créer le modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Entraîner
model.fit(X_train, y_train)

# Prédire sur le test
y_pred = model.predict(X_test)

# Accuracy (pourcentage de bonnes prédictions)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# === 1) Construire un DataFrame "results" avec prix + vrai + prédiction ===

# On récupère les lignes de historical_data correspondant aux dates du set de test
results = historical_data.loc[X_test.index].copy()

# On ajoute la vérité (direction réelle) et la prédiction du modèle
results["y_true"] = y_test
results["y_pred"] = y_pred

# === 2) Tracer le prix de AAPL sur la période de test ===

plt.figure(figsize=(12, 6))
plt.plot(results.index, results["Close"], label="Prix AAPL (réel)")

# === 3) Ajouter les signaux du modèle sur le prix ===

# Points où le modèle prédit une HAUSSE (y_pred = 1)
up_signals = results[results["y_pred"] == 1]
plt.scatter(
    up_signals.index,
    up_signals["Close"],
    marker="^",           # triangle vers le haut
    label="Prédiction Hausse",
    alpha=0.7
)

# Points où le modèle prédit une BAISSE (y_pred = 0)
down_signals = results[results["y_pred"] == 0]
plt.scatter(
    down_signals.index,
    down_signals["Close"],
    marker="v",           # triangle vers le bas
    label="Prédiction Baisse",
    alpha=0.7
)

# === 4) Finitions du graphique ===

plt.title("AAPL – Prix réel + signaux de prédiction du modèle")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
