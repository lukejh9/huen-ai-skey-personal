import sys
from pathlib import Path
import joblib

from src.data_loader import load_dataset


_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BIM_BASE          = _ROOT.parent / "BIM_Data"
DATA_DIR           = _BIM_BASE / "TestModels"
MODEL_DIR          = _ROOT / "models_v2"
PREDICTIONS_DIR    = MODEL_DIR / "predictions"


def predict_data():
    model = joblib.load(MODEL_DIR / "group_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "group_scaler.pkl")
    group_feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    int_to_group = joblib.load(MODEL_DIR / "int_to_group.pkl")

    df = load_dataset(data_dir=DATA_DIR, verbose=True)

    X_new = df[group_feature_cols].fillna(0.0).values
    X_new_s = scaler.transform(X_new)

    pred_labels = model.predict(X_new_s)
    pred_group = [int_to_group.get(i, str(i)) for i in pred_labels]

    df["predicted_group_label"] = pred_labels
    df["predicted_component_group"] = pred_group

    df.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    predict_data()