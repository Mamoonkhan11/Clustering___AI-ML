# Data Loading and Preprocessing Module

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def Load_data(file_path):
    
    # Load dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!\n")
    print(df.head())

    df = df.drop(columns=['CustomerID'], errors='ignore')

    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])

    X = df.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled