import sys
import warnings
from pathlib import Path
import joblib
from sklearn.metrics import classification_report

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

    component_df = df[["source_file", "skey", "predicted_component_group"]]

    component_df.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    predict_valves(df)

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


if __name__ == "__main__":
    predict_data()