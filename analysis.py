import pandas as pd
import random
from datetime import datetime, timedelta

"""
Génère un jeu de données fictif de mouvements de stock pour différents produits agricoles.
- Importe les bibliothèques nécessaires : pandas pour la manipulation de données, random pour la génération aléatoire, datetime pour la gestion des dates.
- Définit les paramètres de génération :
    - Fixe la graine aléatoire pour la reproductibilité.
    - Liste les produits simulés.
    - Liste les types de mouvements de stock (import, export, ajustement, perte).
    - Définit le nombre de lignes à générer.
- Définit les saisons de haute disponibilité pour chaque produit (mois concernés).
- Initialise la liste de données et la date de départ (il y a 10 ans).
- Pour chaque ligne à générer :
    - Tire une date aléatoire sur les 10 dernières années.
    - Sélectionne un produit aléatoirement.
    - Détermine le mois de la date.
    - Calcule dynamiquement les poids de probabilité pour chaque type de mouvement selon la saisonnalité du produit.
    - Normalise les poids pour garantir une somme à 1.
    - Tire un type de mouvement selon les poids calculés.
    - Attribue une quantité selon le type de mouvement et la saisonnalité :
        - Import : quantité positive, plus élevée en saison.
        - Export : quantité négative, plus élevée en saison.
        - Ajustement : quantité variable, plus large en saison.
        - Perte : petite quantité négative.
    - Ajoute la ligne générée à la liste de données.
- Crée un DataFrame pandas à partir des données générées.
- Trie les données par date croissante.
- Exporte le DataFrame au format CSV sous le nom 'mouvements_stock_fictifs.csv'.
"""

# Paramètres
random.seed(42)  # Pour la reproductibilité

products = ['Tomate', 'Carotte', 'Banane', 'Pomme', 'Poivron', 'Courgette', 'Ananas', 'Lait', 'Letchi']
types_movement = ['import', 'export', 'adjustment', 'loss']  # 'return' retiré
n = 10000  # nombre de lignes

# Saison de haute disponibilité par produit (mois: 1=janvier, ..., 12=décembre)
product_seasons = {
    'Letchi': [11, 12],
    'Tomate': [9, 10, 11, 12, 1, 2],
    'Banane': list(range(1, 13)),
    'Carotte': [5, 6, 7, 8, 9],
    'Pomme': [3, 4, 5, 6],
    'Poivron': [10, 11, 12, 1, 2],
    'Courgette': [2, 3, 4, 5],
    'Ananas': [12, 1, 2],
    'Lait': list(range(1, 13)),
}

# Génération des données
data = []
start_date = datetime.today() - timedelta(days=3650)

for _ in range(n):
    date = start_date + timedelta(days=random.randint(0, 3650))
    product = random.choice(products)
    month = date.month

    # Poids dynamiques selon la saison
    if month in product_seasons.get(product, []):
        # Saison haute : plus d'import, un peu plus d'export
        weights = [random.uniform(0.5, 0.65), random.uniform(0.3, 0.4), random.uniform(0.04, 0.08), random.uniform(0.02, 0.06)]
    else:
        # Hors saison : plus d'export, moins d'import
        weights = [random.uniform(0.15, 0.3), random.uniform(0.6, 0.75), random.uniform(0.04, 0.08), random.uniform(0.02, 0.06)]
    # Normalisation des poids
    total = sum(weights)
    weights = [w / total for w in weights]

    movement = random.choices(
        types_movement,
        weights=weights,
        k=1
    )[0]

    # Attribution des quantités selon le type de mouvement
    if movement == 'import':
        quantity = random.randint(200, 700) if month in product_seasons.get(product, []) else random.randint(1, 200)
    elif movement == 'export':
        quantity = -random.randint(200, 700) if month in product_seasons.get(product, []) else -random.randint(1, 200)
    elif movement == 'adjustment':
        quantity = random.randint(-30, 70) if month in product_seasons.get(product, []) else random.randint(-20, 40)
    elif movement == 'loss':
        quantity = -random.randint(1, 30)
    else:
        quantity = 0  # Sécurité

    data.append([date.strftime('%Y-%m-%d'), product, quantity, movement])

# Création et export du DataFrame
df = pd.DataFrame(data, columns=['date', 'product_name', 'quantity', 'type_movement'])
df = df.sort_values(['date'], ascending=[True])
df.to_csv('mouvements_stock_fictifs.csv', index=False)


