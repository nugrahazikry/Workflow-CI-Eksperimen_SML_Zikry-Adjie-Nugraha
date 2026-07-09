"""Training model klasifikasi Churn untuk MLflow Project - Kriteria 3.

Tracking pakai default lokal (folder ./mlruns), TIDAK menunjuk ke server
127.0.0.1 seperti di Kriteria 2, karena ini dijalankan di runner GitHub
Actions yang tidak punya server MLflow yang jalan terus-menerus.
"""

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Set experiment name
mlflow.set_experiment("Telco Customer Churn - Workflow CI")

#Enable autolog
mlflow.sklearn.autolog()

#Load preprocessed dataset (hasil Kriteria 1, disalin ke folder ini)
df = pd.read_csv("namadataset_preprocessing/telco_churn_preprocessed.csv")

#Split fitur (X) dan target (y)
X = df.drop(columns="Churn").to_numpy()
y = df[["Churn"]].to_numpy()
y = y.reshape(len(y),)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Define & train model
with mlflow.start_run():

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Train accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
