import sys
import warnings
from pathlib import Path
import joblib
from sklearn.metrics import classification_report
import pandas as pd

from src.data_loader import load_dataset


_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BIM_BASE          = _ROOT.parent / "BIM_Data"
DATA_DIR           = _BIM_BASE / "TestModels"
MODEL_DIR          = _ROOT / "models_v2"
VALVE_MODEL_DIR    = MODEL_DIR / "valve_models"
PREDICTIONS_DIR    = MODEL_DIR / "predictions"
CSV_PATH           = PREDICTIONS_DIR / "predictions.csv"


def predict_data():
    model = joblib.load(MODEL_DIR / "group_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "group_scaler.pkl")
    group_feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    int_to_group = joblib.load(MODEL_DIR / "int_to_group.pkl")

    df = load_dataset(data_dir=DATA_DIR, verbose=False)

    warnings.filterwarnings("ignore", category=UserWarning)

    X_new = df[group_feature_cols].fillna(0.0).values
    X_new_s = scaler.transform(X_new)

    pred_labels = model.predict(X_new_s)
    pred_group = [int_to_group.get(i, str(i)) for i in pred_labels]

    df["predicted_group_label"] = pred_labels
    df["predicted_component_group"] = pred_group

    #df["skey"] = 

    component_df = df[["source_file", "skey", "predicted_component_group"]]

    component_df.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    valve_df = predict_valves(df)

    valve_df = valve_df.rename(
        columns={"predicted_component_group": "predicted_valve_type"}
    )

    component_df = component_df.merge(
        valve_df[["source_file", "predicted_valve_type"]],
        on="source_file",
        how="left"
    )
    
    report_accuracy(component_df)

def predict_valves(df):
    model = joblib.load(VALVE_MODEL_DIR / "group_classifier.pkl")
    scaler = joblib.load(VALVE_MODEL_DIR / "group_scaler.pkl")
    group_feature_cols = joblib.load(VALVE_MODEL_DIR / "feature_cols.pkl")
    int_to_group = joblib.load(VALVE_MODEL_DIR / "int_to_group.pkl")

    df = df[df["predicted_component_group"] == "VALVE"]

    X_new = df[group_feature_cols].fillna(0.0).values
    X_new_s = scaler.transform(X_new)

    pred_labels = model.predict(X_new_s)
    pred_group = [int_to_group.get(i, str(i)) for i in pred_labels]

    df["predicted_group_label"] = pred_labels
    df["predicted_component_group"] = pred_group

    df = df[["source_file", "skey", "predicted_component_group"]]

    df.to_csv(PREDICTIONS_DIR / "valve_predictions.csv", index=False, encoding="utf-8-sig")

    return df

def report_accuracy(df, verbose=True):
    key = pd.read_csv("source_file_skey.csv")
    total = 0
    correct = 0
    incorrect = []

    for _, row in key.iterrows():
        source = row["source_file"]
        skey = row["Ez-ISO v2 SKEY"]

        data = df[df["source_file"] == source].iloc[0]
        if data["skey"] == skey:
            correct += 1
        else:
            incorrect.append([source, data["skey"], skey])

        total += 1

    if verbose:
        print("")
        print("-" * 65)
        print("Accuracy report:")
        print(f"# Correct:   {correct}")
        print(f"# Incorrect: {len(incorrect)}")
        print(f"# Total:     {total}")
        print(f"Accuracy:    {correct / total}")
        print("")
        print("Incorrect predictions:")
        for i in incorrect:
            print(f"Source: {i[0]}")
            print(f"Predicted SKEY: {i[1]}")
            print(f"Actual SKEY:    {i[2]}")
            print("")

    incorrect_df = pd.DataFrame(
        incorrect,
        columns=["source_file", "predicted_skey", "actual_skey"]
    )

    incorrect_df.to_csv(PREDICTIONS_DIR / "incorrect_skey.csv", index=False)

if __name__ == "__main__":
    predict_data()