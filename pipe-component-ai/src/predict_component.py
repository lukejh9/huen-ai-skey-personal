"""
predict_component.py
---------------------
2단계(stage-2) 추론 스크립트.

  1단계: group_classifier.pkl        → component_group 예측 (VALVE, FITTING, ...)
  2단계: valve_type_classifier.pkl   → component_group == "VALVE" 인 경우에만
                                        valve_type (Ball/Gate/Globe/...) 추가 예측

run_pipeline_v2.py 실행 후 models_v2/ 에 아래 파일들이 있어야 함:
  group_classifier.pkl, feature_cols.pkl, categorical_features.pkl,
  category_values.pkl, int_to_group.pkl
  valve_type_classifier.pkl, valve_feature_cols.pkl,
  valve_categorical_features.pkl, int_to_vtype.pkl
  (밸브 서브셋 표본이 너무 적으면 valve_type_classifier.pkl 은 생성되지
   않을 수 있음 - 이 경우 2단계는 건너뜀)

사용:
    python predict_component.py <json_file_or_dir> [--model-dir models_v2]
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.feature_extractor import extract_features_from_file


class ComponentPredictor:
    """group_classifier + (선택) valve_type_classifier 를 로드해서
    JSON 파일 하나(또는 피처 딕셔너리)로부터 component_group과, VALVE인
    경우 valve_type까지 함께 예측하는 래퍼."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)

        # ── 1단계: component_group ──
        self.group_model = joblib.load(model_dir / "group_classifier.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        self.categorical_cols = joblib.load(model_dir / "categorical_features.pkl")
        self.category_values = joblib.load(model_dir / "category_values.pkl")
        self.int_to_group = joblib.load(model_dir / "int_to_group.pkl")

        # KEYWORD_FEATURES 는 run_pipeline_v2.py Step 1과 동일한 목록이어야
        # feature_cols 순서/구성이 학습 시와 일치한다.
        self.keyword_features = [
            "VALVE", "GASKET", "ELBOW", "TEE", "FLANGE", "OLET",
            "REDUCER", "COUPLING", "CAP", "STRAINER", "SLEEVE",
            "BELLOWS", "FLEXIBLE", "UNION", "BOLT", "WELD",
            "GAUGE", "METER", "SENSOR", "THERMOMETER", "SIGHT",
            "GLASS", "INSTRUMENT", "INDICATOR",
        ]

        # ── 2단계: valve_type (없을 수 있음 - 학습 스킵된 경우) ──
        self.valve_model = None
        valve_pkl = model_dir / "valve_type_classifier.pkl"
        if valve_pkl.exists():
            self.valve_model = joblib.load(valve_pkl)
            self.valve_feature_cols = joblib.load(model_dir / "valve_feature_cols.pkl")
            self.valve_categorical_cols = joblib.load(model_dir / "valve_categorical_features.pkl")
            self.int_to_vtype = joblib.load(model_dir / "int_to_vtype.pkl")

    # ── 피처 딕셔너리 → 학습 시와 동일한 형태의 1행 DataFrame ──
    def _build_feature_row(self, feat: dict) -> pd.DataFrame:
        row = {col: feat.get(col, 0.0) for col in self.feature_cols
               if col not in self.categorical_cols and not col.startswith("kw_")}

        combined_name = (
            str(feat.get("meta_family_name", "")) + " " +
            str(feat.get("meta_type_name", ""))
        ).upper()
        for kw in self.keyword_features:
            col = f"kw_{kw}"
            if col in self.feature_cols:
                row[col] = int(kw in combined_name)

        for col in self.categorical_cols:
            val = feat.get(col, "")
            row[col] = val if val in self.category_values.get(col, []) else ""

        df = pd.DataFrame([row])
        for col in self.categorical_cols:
            df[col] = pd.Categorical(df[col], categories=self.category_values[col])
        return df[self.feature_cols]

    def predict_one(self, json_path: str | Path) -> dict:
        feat = extract_features_from_file(json_path)
        X = self._build_feature_row(feat)

        group_int = int(self.group_model.predict(X)[0])
        group = self.int_to_group.get(group_int, "UNKNOWN")
        group_proba = self.group_model.predict_proba(X)[0]
        group_conf = float(group_proba[group_int])

        result = {
            "source_file": feat.get("source_file", Path(json_path).stem),
            "meta_family_name": feat.get("meta_family_name", ""),
            "meta_type_name": feat.get("meta_type_name", ""),
            "component_group": group,
            "component_group_confidence": round(group_conf, 4),
            "valve_type": None,
            "valve_type_confidence": None,
        }

        # 2단계: VALVE로 예측된 경우에만 세부 타입 추가 예측
        if group == "VALVE" and self.valve_model is not None:
            X_v = X[self.valve_feature_cols]
            vtype_int = int(self.valve_model.predict(X_v)[0])
            vtype_proba = self.valve_model.predict_proba(X_v)[0]
            result["valve_type"] = self.int_to_vtype.get(vtype_int, "UNKNOWN")
            result["valve_type_confidence"] = round(float(vtype_proba[vtype_int]), 4)

        return result

    def predict_many(self, json_paths: list[str | Path]) -> pd.DataFrame:
        rows = []
        for p in json_paths:
            try:
                rows.append(self.predict_one(p))
            except Exception as e:
                rows.append({"source_file": Path(p).stem, "error": str(e)})
        return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="component_group + valve_type 2단계 추론")
    parser.add_argument("path", help="JSON 파일 또는 JSON 파일들이 있는 폴더")
    parser.add_argument("--model-dir", default="models_v2", help="학습된 모델 폴더 (기본: models_v2)")
    parser.add_argument("--out", default=None, help="결과 CSV 저장 경로 (선택)")
    args = parser.parse_args()

    predictor = ComponentPredictor(args.model_dir)

    path = Path(args.path)
    json_files = [path] if path.is_file() else sorted(path.glob("*.json"))
    if not json_files:
        print(f"[오류] JSON 파일을 찾을 수 없습니다: {path}")
        return

    df = predictor.predict_many(json_files)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
