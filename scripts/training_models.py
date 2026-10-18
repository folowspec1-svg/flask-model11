import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATA_FILE = Path("data/diabetes_regression.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "best_regressor_mode.pkl"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
FEATURES_FILE = MODEL_DIR / "feature_names.json"

RANDOM_STATE=42

#load dataset

def load_dataset():
    with DATA_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        #Ambil nama fitur
        feature_names = [
            name for name in reader.fieldnames
            if name != "target"
        ]

        # Read All rows
        rows = list(reader)
    #Ambil data fitur
    X = np.array([
        [
            float(row[feature])
            for feature in feature_names
        ]
        for row in rows
    ])

    # Ambil Target
    y = np.array([
        float(row["target"])
        for row in rows
    ])

    return X, y, feature_names


def build_models():
    model = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=30,
                epsilon=8
            ))
    ]),
        "decision_tree": DecisionTreeRegressor(
            max_depth = 4,
            min_samples_leaf = 10,
            random_state = RANDOM_STATE
        ), 
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "xgbosst": XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE
        )
    }
    return model


def evaluate_model(
    name,
    model,
    x_train,
    x_test,
    y_train,
    y_test
):
    model.fit(x_train, y_train)
  
        # Prediksi
    predictions = model.predict(x_test)

        #evaluasi
    rmse = np.sqrt(
            mean_squared_error(y_test,
            predictions)
        )

    mae = mean_absolute_error(y_test,
        predictions)

    r2 = r2_score(
            y_test,
            predictions
        )

    return {
            "model": name,
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "r2" : round(float(r2), 4)
        }

def main():
    # read dataset
    x,y,feature_names = load_dataset()

    print("Dataset berhasil dibaca")
    print(f"jumlah data : {len(y)}")
    print(f"jumlah fitur : {len(feature_names)}")


    #split data train dan testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size = 0.8, random_state=RANDOM_STATE
    )

    print("Data training: {len{x_train}}")
    print("Data testing: {len(x_test)}")

    #create model
    models = build_models()
    print("Model yang digunakan")

    for name in models:
        print(f"-{name}")

    results = []

    #save best model
    best_model = None
    best_model_name = None
    best_rmse = float("inf")

    for name , model in models.items():
        result = evaluate_model(
            name,
            model,
            x_train,
            x_test,
            y_train,
            y_test)

        results.append(result)

        print(f"{name}: "
            f"RMSE= {result['rmse']},"
            f"MAE= {result['mae']},"
            f"R2= {result['r2']}"
            )

        if result["rmse"] < best_rmse:
            best_rmse = result["rmse"]
            best_model_name = name
            best_model = model

        metadata = {
            "dataset": "Diabete Regression Dataset",
            "task": "regression",
            "feature_name":"feature_names",
            "target":"diabetes diease progression score",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "selection_metric": "lowest-rmse",
            "selected_model": best_model_name,
            "results": sorted(
                results,
                key=lambda item: item["rmse"]
                )
        }

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        #save best model
        joblib.dump(
            best_model,
            MODEL_FILE
        )

        METRICS_FILE.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
        )

        FEATURES_FILE.write_text(
            json.dumps(
                feature_names,
                indent=2
            ),
            encoding="utf-8"
        )

        print("Training Selesai")
        print(f"Model Terbaik {best_model_name}")
        print(f"RMSE Terbail {best_rmse}")
        print(f"Model Disimpan: {MODEL_FILE}")

if __name__ == "__main__":
    main()