import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from skforecast.preprocessing import RollingFeatures
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from skforecast.plot import set_dark_theme

# Les constantes
TEST_SIZE = 24
LAG_SIZE = 6
WINDOW_SIZE = 4

# Charger les données
data = pd.read_csv("dataset/train.csv", delimiter=',', header=0, parse_dates=True, index_col=0)
data = data.sort_index()
data=data.sort_values(by=["item", "store"])

products = data['item'].unique()[:2]
stores = data['store'].unique()[:2]

# Liste des modèles à comparer
models = {
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', eta=0.4, n_estimators=100),
    "RegressionLineaire": LinearRegression(),
    # 'Ridge': Ridge(alpha=1.0),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.2,verbosity=-1),
    'RandomForrest':RandomForestRegressor(random_state=42,criterion="absolute_error")
}
performance={
    "XGBoost":{
        "rmse": [],
        "mae": [],
        "mape": []
    },
    "RegressionLineaire":{
        "rmse": [],
        "mae": [],
        "mape": []
    },
    "LightGBM":{
        "rmse": [],
        "mae": [],
        "mape": []
    },
    "RandomForrest":{
        "rmse": [],
        "mae": [],
        "mape": []
    }
}

for index,product in enumerate(products):
    print(f"Product {index}/{len(products)}")
    for store in stores:
        df = data[(data["item"] == product) & (data["store"] == store)]
        df = df.resample("W").sum().drop(columns=["item", "store"])

        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df = df[["sales", "year", "month", "quarter"]]

        train_set = df[:-TEST_SIZE]
        test_set = df[-TEST_SIZE:]

        for model_name, model in models.items():
            forecaster = ForecasterRecursive(
                regressor=model,
                # steps=TEST_SIZE,
                lags=LAG_SIZE,
                window_features=RollingFeatures(stats=['mean'], window_sizes=WINDOW_SIZE),
                transformer_y=StandardScaler(),
                transformer_exog=StandardScaler()
            )

            forecaster.fit(
                y=train_set["sales"],
                exog=train_set[["year", "month", "quarter"]],
            )

            predictions = forecaster.predict(
                exog=test_set[["year", "month", "quarter"]],
                steps=TEST_SIZE)

            prediction_std = np.std(predictions)
            lower_bound = predictions - 1.96 * prediction_std
            upper_bound = predictions + 1.96 * prediction_std

            # set_dark_theme()
            # fig, ax = plt.subplots(figsize=(15, 5))

            # test_set["sales"].plot(ax=ax, label='Test')
            # predictions.plot(ax=ax, label='Predictions')
            # ax.fill_between(predictions.index, lower_bound, upper_bound, color='gray', alpha=0.2, label='IC 95%')
            # ax.set_title(f"{model_name} - Product {product}, Store {store}", fontsize=14)
            # ax.legend()

            # plt.savefig(f"Forecast_{model_name}_Product-{product}_Store-{store}.png", dpi=300)
            # plt.close()

            # Calculer les métriques
            rmse = np.round(np.sqrt(mean_squared_error(test_set["sales"], np.round(predictions))), 0)
            mae = np.round(mean_absolute_error(test_set["sales"], np.round(predictions)), 0)
            mape = mean_absolute_percentage_error(test_set["sales"], np.round(predictions))
            
            # Stocker les métriques dans la variable performance
            performance[model_name]["rmse"].append(rmse)
            performance[model_name]["mae"].append(mae)
            performance[model_name]["mape"].append(mape)

            # print(f"[{model_name}] Product: {product}, Store: {store}")
            # print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape:.2%}\n")

# Créer un DataFrame pour les métriques moyennes
metrics_df = pd.DataFrame({
    "Model": list(performance.keys()),
    "RMSE_Mean": [np.mean(performance[model]["rmse"]) for model in performance],
    "MAE_Mean": [np.mean(performance[model]["mae"]) for model in performance],
    "MAPE_Mean (en %)": [np.mean(performance[model]["mape"])*100 for model in performance]
})
# On affiche les resultats moyenne des erreurs
print(metrics_df)

# Calculer un score global basé sur une combinaison pondérée des métriques
metrics_df["Score"] = (
    0.5 * metrics_df["RMSE_Mean"] +
    0.3 * metrics_df["MAE_Mean"] +
    0.2 * metrics_df["MAPE_Mean (en %)"]
)

# Identifier le meilleur modèle basé sur le score global
best_model_name = metrics_df.loc[metrics_df["Score"].idxmin(), "Model"]
print("\nMeilleur modèle global basé sur le score pondéré :", best_model_name)

# Charger le modèle sélectionné pour les prédictions futures
best_model = models[best_model_name]

