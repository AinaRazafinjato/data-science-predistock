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
import os


# Liste des modèles à comparer
global models

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "..", "data", "raw", "train copy.csv")



class StockForecaster:
    DEBUG=True
    # DEBUG = True  # Pour activer le mode debug, sinon False
    # Dictionnaire des modèles avec leurs paramètres
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
    
    def __init__(self, 
                 path=file_path, 
                 date_col='date', 
                 product_col='product_id', 
                 target_col='quantity', 
                 mouvement_type_col='movement_type',
                 mouvemement_type=["sale","restock"],
                 model_name="all",
                 horizon=24,
                 frequency='W',
                 start_date=None,
                 test_size=0.2,
                 val_size=0.2,
                 lags=30,
                 window_size=4,
                 ):
        """
        path : chemin du fichier du donnée ou objet Django du donnée
        """
        self.path=path
        self.data = None
        self.freq=frequency
        self.start_date=start_date
        self.end_date=datetime.now()
        self.date_col = date_col
        self.product_col = product_col
        self.target_col = target_col
        self.movement_type_col = mouvement_type_col
        self.exog_col= ["year", "month", "quarter"]
        self.horizon = horizon
        self.test_size = test_size
        self.val_size = val_size
        self.lags = 6
        # self.lags_grid = list(np.arange(start=1,stop=lags))
        self.lags_grid = {f"{lag} lags": int(lag) for lag in np.arange(start=1,stop=lags)}
        self.window_size = window_size
        self.movement_type = mouvemement_type
        self.models = models if model_name=="all" else {model_name:models[model_name],}
        self.best_model = None if model_name=="all" else self.models[model_name]["alg"]
        self.best_params = {name: {} for name in models}
        self.best_lags = {name: None for name in models}
        self.weights = {
                    'RMSE': 0.4,
                    'MAE': 0.3,
                    'MAPE': 0.3
                }
        self.performance= {name: {"rmse": None, "mae": None, "mape": None}
                            for name in self.models} if isinstance(self.models, dict) else {model_name: {"rmse": None, "mae": None, "mape": None}}
                            

    # Chargement des données à partir d'un fichier CSV ou d'une base de données
    def load_data(self):
        """
        Charger les données à partir d'un fichier CSV ou base de donnée.
        """
        # Charger les données
        if isinstance(self.path, str):
            data = pd.read_csv(self.path, delimiter=',', header=0, parse_dates=True, index_col=self.date_col)
            data = data.sort_index()
            data = data.sort_values(by=[self.product_col, self.movement_type_col])
        else:
            # Si c'est un objet Django, on suppose qu'il a une méthode pour récupérer les données
            pass
        
        return data
    
    # Pretaitement du donnee pour l'entrainement du ou des modeles
    def preprocess(self,data):
        # Filtrer, encoder, ajouter features temporelles, etc.
        """
        Crée un DataFrame avec full_range en index et chaque colonne = produit-store.
        Les dates manquantes sont remplies avec 0.
        """
        
        # Copy de la data
        data = data.copy()
        
        # S'assurer que l'index est bien la date
        if self.date_col in data.columns:
            data[self.date_col] = pd.to_datetime(data[self.date_col])
            data = data.set_index(self.date_col)

        # Filtrer par les types de mouvement
        data = data[data[self.movement_type_col].isin(self.movement_type)]
        
        # Créer la clé colonne "produit-mouvement_type"
        data['product_mouvement'] = data[self.product_col].astype(str) + '-' + data[self.movement_type_col].astype(str)

        # Pivot de la data
        pivot_data = data.pivot_table(
            index=self.date_col,
            columns="product_mouvement",
            values=self.target_col,  # <-- à adapter si ta colonne s'appelle différemment
            aggfunc='sum'
        )

        # Reindex sur full_range
        full_range_date = pd.date_range(
                            start=self.start_date if self.start_date!=None else data.index.min(), 
                            end=data.index.max() if StockForecaster.DEBUG else self.end_date, 
                            freq="D")
        pivot_data = pivot_data.reindex(full_range_date)

        # Remplir les valeurs manquantes avec 0
        pivot_data.fillna(0, inplace=True)
        
        # Resample du donnée en frequence voulu
        df = pivot_data.resample(self.freq).sum()
        
        # Ajouter des features temporelles
        df["year"]= df.index.year
        df["month"]= df.index.month
        df["quarter"]= df.index.quarter
        
        self.data=df
        return df
    
    # Separation de données en train, test
    def train_test_split(self,data):
        """ Sépare les données en ensembles d'entraînement et de test.

        Args:
            data (_type_): Un Dataframe contenant les données à séparer.
        test_size (float, optional): Proportion de l'ensemble de données à utiliser pour le test. Defaults to 0.2.

        Returns:
            _type_: Tuple contenant les ensembles d'entraînement et de test, ainsi que les tailles de validation et de test.
        """
        test_size=int(len(data)*self.test_size)
        val_size=int(len(data)*self.val_size)
        train_set_with_exog=data.iloc[:-test_size]
        test_set_with_exog=data.iloc[-test_size:]
        
        return train_set_with_exog, test_set_with_exog,val_size,test_size
    
    # Faire le grid search pour les modele qui ont des params_grid
    def grid_search(self,data):
        train_set, test_set, val_size,test_size = self.train_test_split(data)
        for model_name, model in self.models.items():
            print(f"En cours : {model_name}")
            
            # Initialisation du modèle avec le forecaster
            forecaster = ForecasterRecursiveMultiSeries(
                            regressor = model["alg"],
                            lags      = self.lags,
                            encoding  = 'ordinal',
                            window_features=RollingFeatures(stats=['mean'], window_sizes=self.window_size),
                            transformer_series=StandardScaler(),
                            transformer_exog=StandardScaler(),
                        )
            
            # Si le modele possede un paramètre de grid search, on l'utilise, sinon on l'entraîne directement
            if model["params_grid_search"] is not None:
                
                # Grid Search
                levels = train_set.drop(columns=self.exog_col).columns
                exog=self.exog_col
                
                # Cross-validation
                cv = TimeSeriesFold(
                        steps              = 2,
                        initial_train_size = len(train_set)-val_size,
                        refit              = False
                    )
                
                # Initialisation du grid search
                grid_search = grid_search_forecaster_multiseries(
                            forecaster       = forecaster,
                            series           = train_set.drop(columns=self.exog_col),
                            exog             = train_set[self.exog_col],
                            lags_grid        = self.lags_grid,
                            param_grid       = model["params_grid_search"],
                            cv               = cv,
                            levels           = levels.tolist(),
                            metric           = 'mean_absolute_error',
                            aggregate_metric = 'weighted_average' ,
                            
                        )
                
                # La ligne avec la meilleure combinaison d'hyperparamètres
                best_model_info = grid_search.iloc[0]  

                # # Pour extraire seulement les paramètres
                # best_params = best_model_info['params']
                print(best_model_info)
                # Enregistrement du meilleur params 
                self.best_params[model_name] = best_model_info['params']
                self.best_lags[model_name] = best_model_info['lags']
                
                
                # Mise à jour du modèle avec les meilleurs paramètres
                forecaster.regressor.set_params(**self.best_params[model_name])
            else:
                # Entrainnement sur l'ensemble du train_set et val_set 
                forecaster.fit(
                            series=train_set.drop(columns=self.exog_col),
                            store_in_sample_residuals=True,
                            exog=train_set[self.exog_col],
                            
                            )
                    
            # Prediction sur le test set
            predictions = forecaster.predict(
                        steps=test_size,
                        exog=test_set[self.exog_col],
                    )
            
            predictions.index.name = 'date'
            predictions=predictions.pivot_table(
                        index="date",
                        columns='level',
                        values='pred',
                        aggfunc='sum'
                        )
            
            # Calcul des performances
            metrics_df = self.compute_metrics_per_column(test_set.drop(columns=self.exog_col), predictions)
            
            # Enregistrement des performances
            self.performance[model_name]["rmse"]=np.mean(metrics_df['RMSE'])
            self.performance[model_name]["mae"]=np.mean(metrics_df['MAE'])
            self.performance[model_name]["mape"]=np.mean(metrics_df['MAPE'])
    
    # Recherche du meilleur modele
    def find_best_model(self):
        performance_df = pd.DataFrame(self.performance)
        performance_df = performance_df.T
        performance_df.columns = ['RMSE', 'MAE', 'MAPE']
        
        # Calcul du score pondéré (plus bas = meilleur)
        performance_df['score_pondéré'] = (
            performance_df['RMSE'] * self.weights['RMSE'] +
            performance_df['MAE'] * self.weights['MAE'] +
            performance_df['MAPE'] * self.weights['MAPE']
        )
        # Affichage des performances
        print("Performances des modèles :")
        print(performance_df[['RMSE', 'MAE', 'MAPE', 'score_pondéré']])
        # Tri pour trouver le meilleur modèle
        self.best_model_name = performance_df['score_pondéré'].idxmin()
            
        # Recuperation du meilleur modèle et de ses meilleurs hyperparamètres
        self.best_model=models[self.best_model_name]['alg']
        # self.best_params=self.best_params[self.best_model_name]
        # Utilisation du meilleur parametre sur la meilleur modele
        self.best_model.set_params(**self.best_params[self.best_model_name])
        
        print(performance_df)
    def forecast(self):
        # Generation du future variable exogene
        future_dates= pd.date_range(start=self.data.index.max() + pd.Timedelta(weeks=1), periods=self.horizon, freq=self.freq)
        exog_future = pd.DataFrame({
            "year": future_dates.year,
            "month": future_dates.month,
            "quarter": future_dates.quarter
        },index=future_dates)
        
        # Initialisation du meilleur modèle
        forecaster = ForecasterRecursiveMultiSeries(
                        window_features=RollingFeatures(stats=['mean'], window_sizes=self.window_size),
                        regressor = self.best_model,
                        lags      = self.best_lags[self.best_model_name],
                        encoding  = 'ordinal',
                        transformer_series=StandardScaler(),
                        transformer_exog=StandardScaler(),    
                        )
        
        forecaster.fit(
                    series=self.data.drop(columns=self.exog_col),
                    store_in_sample_residuals=True,
                    exog=self.data[self.exog_col],
                    )
        
        # Prediction future
        predictions = forecaster.predict_interval(
                    steps=self.horizon,
                    exog=exog_future,
                    )
        print("Les meilleur lags par modele ",self.best_lags)
        print(forecaster.summary())
        return predictions
    # Evaluation des performances du modèle
    def compute_metrics_per_column(self,y_true, y_pred):
        metrics = {}
        for col in y_true.columns:
            rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
            mae = mean_absolute_error(y_true[col], y_pred[col])
            mape = mean_absolute_percentage_error(y_true[col], y_pred[col]) * 100
            metrics[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        return pd.DataFrame(metrics).T  # transpose for readability
     
    # Exécution de l'ensemble du processus
    def run_all(self):
        data=self.load_data()
        data_processed=self.preprocess(data)
        self.grid_search(data_processed)
        self.find_best_model()
        print(self.forecast())
        return data_processed


if __name__=="__main__":
    ex=StockForecaster(
                    mouvemement_type=[1,2],
                    lags=2
                    )
    
    ex.run_all()