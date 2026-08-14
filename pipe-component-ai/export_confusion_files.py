import pandas as pd

LABEL_COLUMN = "component_group"
PRED_COLUMN = "pred_component_group"
df = pd.read_csv(
    "models_v2/dataset.csv"
)


print("Columns:")
print(df.columns.tolist())


# CHANGE THIS after checking your columns
LABEL_COLUMN = "component_group"


pairs = [
    ("COUPLING", "OLET"),
    ("JOINT", "VALVE"),
    ("JOINT", "COUPLING"),
]


for true_label, pred_label in pairs:

    subset = df[
        (df["component_group"] == "component_group") &
        (df["pred_component_group"] == "pred_component_group")
    ]


    print(
        true_label,
        "samples:",
        len(subset)
    )


    subset.to_csv(
        f"models_v2/check_{true_label}_samples.csv",
        index=False
    )


print("Done")