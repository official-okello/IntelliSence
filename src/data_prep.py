import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customer_data = 'data/customers.csv'
transactions_data = 'data/transactions.csv'

def customer_data(path: str):
    return pd.read_csv(customer_data)

def transactions_data(path: str):
    return pd.read_csv(transactions_data)

def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test