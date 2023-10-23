# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('data/data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'model/model.pkl')

X_test.to_csv('data/test.csv', index=False)
y_test.to_csv('data/test_labels.csv', index=False)
