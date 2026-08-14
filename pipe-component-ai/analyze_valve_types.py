import pandas as pd

df = pd.read_csv("models_v2/dataset.csv")

valves = df[df["is_valve"] == True]

print(
    valves["valve_type"]
    .value_counts()
)

print("\nMissing:")
print(
    valves[valves["valve_type"].isna()]
)