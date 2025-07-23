import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from skforecast.preprocessing import RollingFeatures
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import grid_search_forecaster
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

# Constantes
TEST_SIZE = 24
VAL_SIZE = 12
LAG_SIZE = 6
WINDOW_SIZE = 4

# Charger les données
data = pd.read_csv("data/raw/mouvements_stock_fictifs.csv", delimiter=',', header=0, parse_dates=True, index_col=0)
data = data.sort_index()
data = data.sort_values(by=["item", "store"])

products = data['item'].unique()[:2]
stores = data['store'].unique()[:2]

# Liste des modèles à comparer
models = {
    'XGBoost': {
        "alg": xgb.XGBRegressor(objective='reg:squarederror', verbosity=0),
        "params_grid_search": {
            "eta": [0.1, 0.3],
            "n_estimators": [50, 100]
        }
    },
    "RegressionLineaire": {
        "alg": LinearRegression(),
        "params_grid_search": None
    },
    'Ridge': {
        "alg": Ridge(alpha=1.0),
        "params_grid_search": {
            "alpha": [0.1, 0.5, 1.0]
        }
    },
    'LightGBM': {
        "alg": LGBMRegressor(n_estimators=100, learning_rate=0.2, verbosity=-1),
        "params_grid_search": {
            "learning_rate": [0.1, 0.2],
            "n_estimators": [50, 100]
        }
    },
    'RandomForrest': {
        "alg": RandomForestRegressor(random_state=42, criterion="absolute_error"),
        "params_grid_search": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20]
        }
    }
}

# Pour stocker les performances
performance = {
    name: {"rmse": [], "mae": [], "mape": []}
    for name in models
}
best_parameters = {name: {} for name in models}

# Boucle produit-store
for index, product in enumerate(products):
    print(f"Product {index + 1}/{len(products)}")
    for store in stores:
        df = data[(data["item"] == product) & (data["store"] == store)]
        df = df.resample("W").sum().drop(columns=["item", "store"])

        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df = df[["sales", "year", "month", "quarter"]]

        train_set = df[:-(TEST_SIZE + VAL_SIZE)]
        test_set = df[-TEST_SIZE:]

        for model_name, model in models.items():
            print(f"  → Model: {model_name} | Product: {product} | Store: {store}")
            
            forecaster = ForecasterRecursive(
                regressor=model["alg"],
                lags=LAG_SIZE,
                window_features=RollingFeatures(stats=['mean'], window_sizes=WINDOW_SIZE),
                transformer_y=StandardScaler(),
                transformer_exog=StandardScaler()
            )

            # Grid Search si applicable
            if model["params_grid_search"]:
                results = grid_search_forecaster(
                    forecaster=forecaster,
                    y=train_set["sales"],
                    exog=train_set[["year", "month", "quarter"]],
                    param_grid=model["params_grid_search"],
                    metric='mean_squared_error',
                    # steps=TEST_SIZE,
                    cv=3,
                    refit=True,
                    return_best=True,
                    verbose=0
                )
                forecaster = results.forecaster
                best_parameters[model_name] = results.best_params
            else:
                forecaster.fit(
                    y=train_set["sales"],
                    exog=train_set[["year", "month", "quarter"]],
                )

            # Prédiction
            predictions = forecaster.predict(
                steps=TEST_SIZE,
                exog=test_set[["year", "month", "quarter"]]
            )

            # Évaluation
            rmse = np.round(np.sqrt(mean_squared_error(test_set["sales"], predictions)), 0)
            mae = np.round(mean_absolute_error(test_set["sales"], predictions), 0)
            mape = mean_absolute_percentage_error(test_set["sales"], predictions)

            performance[model_name]["rmse"].append(rmse)
            performance[model_name]["mae"].append(mae)
            performance[model_name]["mape"].append(mape)

# Affichage des résultats
print("\n=== Meilleures performances ===")
for model_name, scores in performance.items():
    print(f"\n{model_name}:")
    print(f"  RMSE Moyenne: {np.mean(scores['rmse']):.2f}")
    print(f"  MAE Moyenne:  {np.mean(scores['mae']):.2f}")
    print(f"  MAPE Moyenne: {np.mean(scores['mape'])*100:.2f}%")

print("\n=== Meilleurs hyperparamètres trouvés ===")
for model_name, params in best_parameters.items():
    print(f"{model_name}: {params}")
