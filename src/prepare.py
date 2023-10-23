# prepare.py

from sklearn.datasets import make_classification
import pandas as pd

data, target = make_classification(n_samples=200, n_features=20, random_state=42)
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(20)])
df['target'] = target

df.to_csv('data/data.csv', index=False)
