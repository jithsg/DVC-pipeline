# evaluate.py
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

model = joblib.load('model/model.pkl')

X_test = pd.read_csv('data/test.csv')
y_test = pd.read_csv('data/test_labels.csv')

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

metrics = {
    'accuracy': accuracy
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
