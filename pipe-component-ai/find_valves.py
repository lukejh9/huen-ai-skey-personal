import pandas as pd


df = pd.read_csv(
    "models_v2/dataset.csv"
)


# all valve group samples
valves = df[
    df["component_group"] == "VALVE"
]


print(
    "VALVE GROUP:",
    len(valves)
)


print(
    valves[
        [
            "source_file",
            "meta_family_name",
            "meta_type_name",
            "valve_type"
        ]
    ]
    .to_string()
)


valves.to_csv(
    "models_v2/all_valves_review.csv",
    index=False,
    encoding="utf-8-sig"
)