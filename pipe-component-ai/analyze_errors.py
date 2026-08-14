import pandas as pd
from pathlib import Path


MODEL_DIR = Path("models_v2")


report = pd.read_csv(
    MODEL_DIR / "classification_report.csv",
    index_col=0
)


cm = pd.read_csv(
    MODEL_DIR / "confusion_matrix.csv",
    index_col=0
)


print("\n=== Worst Classes ===")


for cls in report.index:

    cls_name = str(cls)

    if cls_name in [
        "accuracy",
        "macro avg",
        "weighted avg"
    ]:
        continue

    f1 = float(
        report.loc[cls, "f1-score"]
    )

    if f1 < 0.90:
        print(
            f"{cls_name:20s} F1={f1:.3f}"
        )


print("\n=== Confusions ===")


for true_cls in cm.index:

    true_name = str(true_cls)

    row = cm.loc[true_cls]


    for pred_cls, count in row.items():

        pred_name = str(pred_cls)

        if (
            pred_name != true_name
            and int(count) > 0
        ):
            print(
                f"{true_name:20s} -> {pred_name:20s}: {count}"
            )