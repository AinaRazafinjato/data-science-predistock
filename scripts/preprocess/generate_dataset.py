import pandas as pd
import random
from datetime import datetime, timedelta

def get_product_seasons():
    return {
        'Lait': list(range(1, 13)),
        'Banane': list(range(1, 13)),
        'Letchi': [11, 12],
        'Tomate': [9, 10, 11, 12, 1, 2],
    }

def get_weights(month, product, product_seasons):
    if month in product_seasons.get(product, []):
        weights = [
            random.uniform(0.5, 0.65),  # import
            random.uniform(0.3, 0.4),   # export
            random.uniform(0.04, 0.08), # adjustment
            random.uniform(0.02, 0.06)  # loss
        ]
    else:
        weights = [
            random.uniform(0.15, 0.3),  # import
            random.uniform(0.6, 0.75),  # export
            random.uniform(0.04, 0.08), # adjustment
            random.uniform(0.02, 0.06)  # loss
        ]
    total = sum(weights)
    return [w / total for w in weights]

def get_quantity(movement, month, product, product_seasons):
    if movement == 'import':
        return random.randint(200, 700) if month in product_seasons.get(product, []) else random.randint(1, 200)
    elif movement == 'export':
        return -random.randint(200, 700) if month in product_seasons.get(product, []) else -random.randint(1, 200)
    elif movement == 'adjustment':
        return random.randint(-30, 70) if month in product_seasons.get(product, []) else random.randint(-20, 40)
    elif movement == 'loss':
        return -random.randint(1, 30)
    return 0

def generate_data(n, products, types_movement, product_seasons):
    random.seed(42)
    data = []
    start_date = datetime.today() - timedelta(days=3650)
    for _ in range(n):
        date = start_date + timedelta(days=random.randint(0, 3650))
        product = random.choice(products)
        month = date.month
        if month not in product_seasons[product]:
            continue
        weights = get_weights(month, product, product_seasons)
        movement = random.choices(types_movement, weights=weights, k=1)[0]
        quantity = get_quantity(movement, month, product, product_seasons)
        data.append([date.strftime('%Y-%m-%d'), product, quantity, movement])
    return data

def main():
    products = ['Tomate', 'Banane', 'Lait', 'Letchi']
    types_movement = ['import', 'export', 'adjustment', 'loss']
    n = 10000
    product_seasons = get_product_seasons()
    data = generate_data(n, products, types_movement, product_seasons)
    df = pd.DataFrame(data, columns=['date', 'product_name', 'quantity', 'type_movement'])
    df = df.sort_values(['date'], ascending=True)
    df.to_csv('../../data/raw/mouvements_stock_fictifs.csv', index=False)

if __name__ == "__main__":
    main()
