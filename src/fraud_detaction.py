from sklearn.ensemble import IsolationForest

def train_fraud_model(X):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    return model