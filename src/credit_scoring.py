from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_credit_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model