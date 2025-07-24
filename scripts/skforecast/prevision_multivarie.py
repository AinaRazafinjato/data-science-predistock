# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from skforecast.datasets import fetch_dataset

# Importation des modeles
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold,
    grid_search_forecaster_multiseries,
)
from datetime import datetime


# Liste des modèles à comparer
global models
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
    "Ridge": {
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
    # 'RandomForrest': {
    #     "alg": RandomForestRegressor(random_state=42, criterion="absolute_error"),
    #     "params_grid_search": {
    #         "n_estimators": [50, 100],
    #         "max_depth": [None, 10, 20]
    #     }
    # }
}

class StockForecaster:
    
    def __init__(self, path="../../data/raw/train.csv", 
                 date_col='date', 
                 product_col='product_id', 
                 target_col='quantity', 
                 mouvemement_type=["sale","restock"],
                 model_name="all",
                 horizon=12,
                 frequency='M',
                 start_date=None,
                 ):
        """
        path : chemin du fichier du donnée ou objet Django du donnée
        """
        self.path=path
        self.data = None
        self.freq=frequency
        self.start_date=start_date
        self.date_col = date_col
        self.product_col = product_col
        self.target_col = target_col
        self.movement_type = mouvemement_type
        self.models = models if model_name=="all" else models[model_name]
        self.best_model = None if model_name=="all" else self.models["alg"]
        self.performance= {name: {"rmse": None, "mae": None, "mape": None}
                            for name in self.models} if isinstance(self.models, dict) else {model_name: {"rmse": None, "mae": None, "mape": None}}
                            

        
    def load_data(self):
        """
        Charger les données à partir d'un fichier CSV ou base de donnée.
        """
        # Charger les données
        if isinstance(self.path, str):
            data = pd.read_csv(self.path, delimiter=',', header=0, parse_dates=True, index_col=self.date_col)
            data = data.sort_index()
            data = data.sort_values(by=["item", "store"])
        else:
            # Si c'est un objet Django, on suppose qu'il a une méthode pour récupérer les données
            pass
        return data
    
    
    
    # Pretaitement du donnee pour l'entrainement du ou des modeles
    def preprocess(self):
        # Filtrer, encoder, ajouter features temporelles, etc.
        """
        Crée un DataFrame avec full_range en index et chaque colonne = produit-store.
        Les dates manquantes sont remplies avec 0.
        """
        # Convertir en datetime
        data = self.load_data().copy()
        
        # S'assurer que l'index est bien la date
        if self.date_col in data.columns:
            data[self.date_col] = pd.to_datetime(data[self.date_col])
            data = data.set_index(self.date_col)

        # Créer la clé colonne "produit-mouvement_type"
        data['produit_mouvement'] = data[self.product_col].astype(str) + '-' + data[self.movement_type].astype(str)

        # Pivot de la data
        pivot_data = data.pivot_table(
            index=self.date_col,
            columns='produit_store',
            values=self.target_col,  # <-- à adapter si ta colonne s'appelle différemment
            aggfunc='sum'
        )

        # Reindex sur full_range
        full_range = pd.date_range(start=self.start_date if self.start_date!=None else data.index.max(), end=datetime.now, freq=self.freq)
        pivot = pivot.reindex(full_range)

        # Remplir les valeurs manquantes avec 0
        pivot.fillna(0, inplace=True)

        return pivot
    
    
    
    
    
    
    # Recuperation du modele a utilisee
    def get_model(self):
        return self.models
        

    # def train(self):
        

    # def predict(self, future_data):
        

    # def run_all(self):
    #     self.preprocess()
    #     self.train()
    #     # Tu peux ajouter un save_model() ou predict() ici


if __name__=="__main__":
    ex=StockForecaster(
                    model_name="Ridge",
                    date_col='date', 
                    product_col='product', 
                    target_col='sales', 
                    mouvemement_type=["1","2"],
                    horizon=12,
                    frequency='W'
                    )
    print(ex.preprocess().head())