import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

@dataclass
class ProductInfo:
    name: str
    shelf_life_days: int  # durée de conservation en jours
    peak_seasons: List[int]  # mois de haute saison
    base_demand: int  # demande de base
    volatility: float  # volatilité des prix/quantités
    storage_cost_per_day: float  # coût de stockage par jour
    min_stock_threshold: int  # seuil minimum de stock
    max_stock_threshold: int  # seuil maximum de stock
    
def get_product_catalog():
    """Définit les caractéristiques réalistes des produits pour Madagascar"""
    return {
        'Tomate': ProductInfo('Tomate', 14, [4, 5, 6, 7, 8, 9], 400, 0.3, 0.5, 50, 800),
        'Banane': ProductInfo('Banane', 7, list(range(1, 13)), 600, 0.25, 0.3, 100, 1200),
        'Lait': ProductInfo('Lait', 5, list(range(1, 13)), 800, 0.15, 1.0, 150, 1500),
        'Letchi': ProductInfo('Letchi', 10, [11, 12], 200, 0.5, 0.8, 0, 500)
    }

def get_market_trends():
    """Tendances du marché par mois - climat tropical de Madagascar"""
    return {
        1: 1.3, 2: 1.35, 3: 1.4, 4: 1.2, 5: 1.0, 6: 0.9,
        7: 0.85, 8: 0.8, 9: 0.9, 10: 1.1, 11: 1.25, 12: 1.3
    }

def calculate_seasonal_factor(month: int, product_info: ProductInfo, market_trends: Dict) -> float:
    """Calcule le facteur saisonnier combiné pour Madagascar"""
    base_factor = market_trends[month]
    
    # Gestion spéciale pour les letchis - disponibles SEULEMENT en nov-déc
    if product_info.name == 'Letchi':
        if month in [11, 12]:
            return base_factor * 2.0
        else:
            return 0.0
    
    # Pour les autres produits
    if month in product_info.peak_seasons:
        seasonal_bonus = 1.3
    else:
        seasonal_bonus = 0.8
    
    return base_factor * seasonal_bonus

class StockManager:
    """Gestionnaire de stock qui garantit des mouvements réalistes"""
    
    def __init__(self, product_catalog: Dict[str, ProductInfo]):
        self.product_catalog = product_catalog
        self.current_stocks = {}
        self.stock_ages = {}
        self.fournisseurs = [f"Fournisseur_{i}" for i in range(1, 6)]
        self.clients = [f"Client_{i}" for i in range(1, 16)]
        self.fournisseur_idx = 0
        self.client_idx = 0
        
        # Initialisation des stocks
        for product_name, product_info in product_catalog.items():
            if product_name == 'Letchi':
                self.current_stocks[product_name] = 0
                self.stock_ages[product_name] = []
            else:
                initial_stock = random.randint(product_info.min_stock_threshold, 
                                             product_info.max_stock_threshold // 2)
                self.current_stocks[product_name] = initial_stock
                self.stock_ages[product_name] = [(initial_stock, random.randint(1, 3))]
    
    def get_current_stock(self, product_name: str) -> int:
        """Retourne le stock actuel d'un produit"""
        return sum(qty for qty, age in self.stock_ages[product_name])
    
    def age_stock(self, product_name: str):
        """Vieillit le stock d'un jour"""
        self.stock_ages[product_name] = [(qty, age + 1) for qty, age in self.stock_ages[product_name]]
    
    def add_stock(self, product_name: str, quantity: int):
        """Ajoute du stock (import)"""
        if quantity > 0:
            self.stock_ages[product_name].append((quantity, 0))
    
    def remove_stock(self, product_name: str, quantity: int) -> int:
        """Retire du stock (export) - FIFO, retourne la quantité réellement retirée"""
        if quantity <= 0:
            return 0
        
        current_stock = self.get_current_stock(product_name)
        actual_quantity = min(quantity, current_stock)
        
        if actual_quantity == 0:
            return 0
        
        # Retirer du stock (FIFO - First In, First Out)
        remaining_to_remove = actual_quantity
        new_stock_ages = []
        
        # Trier par âge (plus ancien en premier)
        sorted_stock = sorted(self.stock_ages[product_name], key=lambda x: x[1], reverse=True)
        
        for qty, age in sorted_stock:
            if remaining_to_remove <= 0:
                new_stock_ages.append((qty, age))
            elif qty <= remaining_to_remove:
                remaining_to_remove -= qty
            else:
                new_stock_ages.append((qty - remaining_to_remove, age))
                remaining_to_remove = 0
        
        self.stock_ages[product_name] = new_stock_ages
        return actual_quantity
    
    def calculate_losses(self, product_name: str) -> int:
        """Calcule les pertes dues à la périssabilité"""
        product_info = self.product_catalog[product_name]
        losses = 0
        new_stock_ages = []
        
        for qty, age in self.stock_ages[product_name]:
            if age >= product_info.shelf_life_days:
                losses += qty
            else:
                new_stock_ages.append((qty, age))
        
        self.stock_ages[product_name] = new_stock_ages
        return losses
    
    def should_import(self, product_name: str, daily_demand: int) -> bool:
        """Détermine s'il faut importer"""
        product_info = self.product_catalog[product_name]
        current_stock = self.get_current_stock(product_name)
        
        # Import obligatoire si stock insuffisant pour la demande
        if current_stock < daily_demand:
            return True
        
        # Import préventif si stock sous le seuil minimum
        if current_stock < product_info.min_stock_threshold:
            return True
        
        # Import aléatoire pour maintenir un stock sain
        if current_stock < product_info.max_stock_threshold * 0.3 and random.random() < 0.3:
            return True
        
        return False
    
    def calculate_import_quantity(self, product_name: str, daily_demand: int) -> int:
        """Calcule la quantité à importer"""
        product_info = self.product_catalog[product_name]
        current_stock = self.get_current_stock(product_name)
        
        # Quantité basée sur la demande et les seuils
        target_stock = min(product_info.max_stock_threshold, 
                          max(daily_demand * 7, product_info.min_stock_threshold * 2))
        
        import_quantity = max(0, target_stock - current_stock)
        
        # Ajouter de la variabilité
        variability = random.uniform(0.8, 1.4)
        import_quantity = int(import_quantity * variability)
        
        return max(50, import_quantity)  # Minimum 50 unités

def generate_realistic_movements(start_date: datetime, end_date: datetime, 
                               product_catalog: Dict[str, ProductInfo]) -> List[Dict]:
    """Génère des mouvements réalistes avec contraintes strictes"""
    
    movements = []
    stock_manager = StockManager(product_catalog)
    market_trends = get_market_trends()
    current_date = start_date
    
    while current_date <= end_date:
        is_inventory_day = current_date.day == 1 and current_date.month in [1, 4, 7, 10]
        
        for product_name, product_info in product_catalog.items():
            # Vieillissement du stock
            stock_manager.age_stock(product_name)
            
            # Gestion des pertes (périssabilité)
            losses = stock_manager.calculate_losses(product_name)
            if losses > 0:
                movements.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'product_name': product_name,
                    'quantity': -losses,
                    'type_movement': 'adjustment',
                    'reason': 'perishable_loss',
                    'stock_level': stock_manager.get_current_stock(product_name),
                    'partner': 'Internal'
                })
            
            # Calcul de la demande
            seasonal_factor = calculate_seasonal_factor(current_date.month, product_info, market_trends)
            
            if product_info.name == 'Letchi' and seasonal_factor == 0.0:
                daily_demand = 0
            else:
                daily_demand = int(product_info.base_demand * seasonal_factor * 
                                 (1 + np.random.normal(0, product_info.volatility)) / 30)
                daily_demand = max(0, daily_demand)
            
            current_stock = stock_manager.get_current_stock(product_name)
            
            # PRIORITÉ 1: Import si nécessaire (AVANT tout export)
            if stock_manager.should_import(product_name, daily_demand):
                import_quantity = stock_manager.calculate_import_quantity(product_name, daily_demand)
                
                movements.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'product_name': product_name,
                    'quantity': import_quantity,
                    'type_movement': 'import',
                    'reason': 'restock' if current_stock < product_info.min_stock_threshold else 'preventive',
                    'stock_level': current_stock,
                    'partner': stock_manager.fournisseurs[stock_manager.fournisseur_idx % len(stock_manager.fournisseurs)]
                })
                
                stock_manager.add_stock(product_name, import_quantity)
                stock_manager.fournisseur_idx += 1
                current_stock = stock_manager.get_current_stock(product_name)
            
            # PRIORITÉ 2: Export seulement si stock suffisant
            if current_stock > 0 and daily_demand > 0:
                # Export limité au stock disponible et à la demande
                max_export = min(current_stock, daily_demand)
                export_quantity = int(max_export * random.uniform(0.7, 1.0))
                export_quantity = min(export_quantity, stock_manager.get_current_stock(product_name))  # <-- AJOUT
                
                if export_quantity > 0:
                    actual_export = stock_manager.remove_stock(product_name, export_quantity)
                    
                    if actual_export > 0:
                        movements.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'product_name': product_name,
                            'quantity': -actual_export,
                            'type_movement': 'export',
                            'reason': 'normal_sale',
                            'stock_level': current_stock,
                            'partner': stock_manager.clients[stock_manager.client_idx % len(stock_manager.clients)]
                        })
                        
                        stock_manager.client_idx += 1
            
            # PRIORITÉ 3: Ajustements d'inventaire (rares)
            if is_inventory_day and random.random() < 0.8:
                current_stock = stock_manager.get_current_stock(product_name)
                # Ajustement limité pour éviter les stocks négatifs
                max_negative_adjustment = -min(current_stock, current_stock // 4)
                adjustment = random.randint(max_negative_adjustment, 50)
                
                if adjustment != 0:
                    if adjustment > 0:
                        stock_manager.add_stock(product_name, adjustment)
                    else:
                        actual_adjustment = -stock_manager.remove_stock(product_name, min(-adjustment, stock_manager.get_current_stock(product_name)))  # <-- AJOUT
                        adjustment = actual_adjustment
                    
                    if adjustment != 0:
                        movements.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'product_name': product_name,
                            'quantity': adjustment,
                            'type_movement': 'adjustment',
                            'reason': 'inventory_correction',
                            'stock_level': current_stock,
                            'partner': 'Internal'
                        })
        
        current_date += timedelta(days=1)
    
    return movements

def add_market_data(movements: List[Dict]) -> List[Dict]:
    """Ajoute les données de marché (prix, valeurs)"""
    enhanced_movements = []
    base_prices = {'Tomate': 2.5, 'Banane': 1.8, 'Lait': 1.2, 'Letchi': 8.0}
    
    for movement in movements:
        enhanced_movement = movement.copy()
        product = movement['product_name']
        base_price = base_prices.get(product, 2.0)
        
        # Facteur de prix selon le type de mouvement
        if movement['type_movement'] == 'import':
            price_factor = random.uniform(0.8, 1.0)  # Prix d'achat plus bas
        elif movement['type_movement'] == 'export':
            price_factor = random.uniform(1.0, 1.3)  # Prix de vente plus élevé
        elif movement.get('reason') == 'perishable_loss':
            price_factor = 0.0  # Pas de valeur pour les pertes
        else:
            price_factor = 1.0
        
        enhanced_movement['unit_price'] = round(base_price * price_factor, 2)
        enhanced_movement['total_value'] = round(abs(movement['quantity']) * enhanced_movement['unit_price'], 2)
        
        enhanced_movements.append(enhanced_movement)
    
    return enhanced_movements

def generate_realistic_dataset(n_days: int = 365) -> pd.DataFrame:
    """Génère un dataset réaliste sur n_days jours"""
    
    # Dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_days)
    
    # Générer les mouvements
    product_catalog = get_product_catalog()
    movements = generate_realistic_movements(
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.min.time()),
        product_catalog
    )
    
    # Ajouter les données de marché
    enhanced_movements = add_market_data(movements)
    
    # Créer le DataFrame
    df = pd.DataFrame(enhanced_movements)
    
    if not df.empty:
        # Trier par date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'product_name']).reset_index(drop=True)
        
        # Ajouter des colonnes calculées
        df['week'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        # Calculer le stock cumulé par produit (JAMAIS NÉGATIF)
        df['cumulative_stock'] = 0
        for product in df['product_name'].unique():
            mask = df['product_name'] == product
            product_df = df.loc[mask].copy()
            product_df['quantity'] = pd.to_numeric(product_df['quantity'], errors='coerce').fillna(0)
            
            # Correction: recalculer le stock cumulé sans jamais passer sous zéro
            cumulative = []
            stock = 0
            for qty in product_df['quantity']:
                stock += qty
                stock = max(stock, 0)  # Jamais négatif
                cumulative.append(stock)
            df.loc[mask, 'cumulative_stock'] = cumulative

            min_stock = min(cumulative) if cumulative else 0
            if min_stock < 0:
                print(f"ATTENTION: Stock négatif détecté pour {product}: {min_stock}")
        
        # Ajouter l'âge moyen du stock
        df['days_in_stock'] = df.apply(lambda row: random.randint(0, 
            get_product_catalog()[row['product_name']].shelf_life_days // 2), axis=1)
    
    return df

def analyze_dataset_balance(df: pd.DataFrame) -> Dict:
    """Analyse l'équilibre import/export du dataset"""
    analysis = {}
    
    for product in df['product_name'].unique():
        product_data = df[df['product_name'] == product].copy()
        
        # S'assurer que les quantités sont numériques
        product_data['quantity'] = pd.to_numeric(product_data['quantity'], errors='coerce').fillna(0)
        
        imports = product_data[product_data['type_movement'] == 'import']['quantity'].sum()
        exports = abs(product_data[product_data['type_movement'] == 'export']['quantity'].sum())
        adjustments = product_data[product_data['type_movement'] == 'adjustment']['quantity'].sum()
        
        # S'assurer que cumulative_stock est numérique
        if 'cumulative_stock' in product_data.columns:
            product_data['cumulative_stock'] = pd.to_numeric(product_data['cumulative_stock'], errors='coerce').fillna(0)
            final_stock = product_data['cumulative_stock'].iloc[-1] if len(product_data) > 0 else 0
            min_stock = product_data['cumulative_stock'].min()
        else:
            final_stock = 0
            min_stock = 0
        
        analysis[product] = {
            'imports': int(imports),
            'exports': int(exports),
            'adjustments': int(adjustments),
            'balance': int(imports - exports + adjustments),
            'import_export_ratio': float(imports / exports) if exports > 0 else float('inf'),
            'final_stock': int(final_stock),
            'min_stock': int(min_stock),
            'movements_count': len(product_data)
        }
    
    return analysis

def main():
    """Fonction principale"""
    print("Génération d'un dataset réaliste et équilibré de mouvements de stock...")
    
    # Générer le dataset
    df = generate_realistic_dataset(n_days=730)
    
    print(f"Dataset généré avec {len(df)} mouvements")
    print(f"Période: {df['date'].min()} à {df['date'].max()}")
    print(f"Produits: {list(df['product_name'].unique())}")
    
    # Analyse de l'équilibre
    movement_counts = df['type_movement'].value_counts()
    print(f"\nRépartition des mouvements:")
    for movement_type, count in movement_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {movement_type}: {count} ({percentage:.1f}%)")
    
    # Analyse détaillée par produit
    analysis = analyze_dataset_balance(df)
    print(f"\n--- Analyse par produit ---")
    
    for product, data in analysis.items():
        print(f"\n{product}:")
        print(f"  Imports: {data['imports']}")
        print(f"  Exports: {data['exports']}")
        print(f"  Ratio Import/Export: {data['import_export_ratio']:.2f}")
        print(f"  Stock final: {data['final_stock']}")
        print(f"  Stock minimum: {data['min_stock']}")
        print(f"  Mouvements: {data['movements_count']}")
        
        if data['min_stock'] < 0:
            print(f"  ⚠️  PROBLÈME: Stock négatif détecté!")
        else:
            print(f"  ✅ Stock toujours positif")
    
    # Sauvegarder
    output_file = 'mouvements_stock_equilibres.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset sauvegardé dans {output_file}")
    
    return df

if __name__ == "__main__":
    dataset = main()