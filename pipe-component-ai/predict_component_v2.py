"""
predict_component_v2.py
------------------------
3단계 추론 스크립트 (predict_component.py 확장판).

  1단계: group_classifier.pkl        → component_group 예측 (VALVE, FITTING, ...)
  2단계: valve_type_classifier.pkl   → component_group == "VALVE" 인 경우에만
                                        valve_type (Ball/Gate/Globe/...) 추가 예측
  3단계: skey_classifier.pkl         → component_group == "VALVE" 인 경우에만
                                        개별 SKEY 접두어 (VB/BVA/VT/KG/...) 추가 예측
                                        (skey_classifier.pkl 이 없으면 자동으로 건너뜀 -
                                         run_pipeline_v2.py Step 5가 표본 부족으로
                                         스킵된 경우 대비)

predict_component.py 대비 변경점:
  - path.glob("*.json") → path.rglob("*.json") : 하위 폴더까지 재귀 탐색
    (BIM_Data/RealData/{Equipment,Inline_Instrument,NestedFamily,Revit2025}/
     처럼 여러 서브폴더에 JSON이 나뉘어 있는 실사용 데이터 대응)
  - 결과에 "source_subfolder" 컬럼 추가 (입력 폴더 기준 상대 경로의 최상위
    폴더명 - 어느 카테고리 폴더에서 나온 예측인지 감사용으로 남김)
  - confidence 기반 review_required / review_reason 추가 (predict.py에 있던
    0.9(auto)/0.7(review) 규칙을 이식 - group과, VALVE인 경우 valve_type/skey
    신뢰도까지 반영)

run_pipeline_v2.py 실행 후 models_v2/ 에 아래 파일들이 있어야 함:
  group_classifier.pkl, feature_cols.pkl, categorical_features.pkl,
  category_values.pkl, int_to_group.pkl
  valve_type_classifier.pkl, valve_feature_cols.pkl,
  valve_categorical_features.pkl, int_to_vtype.pkl   (없으면 stage 2 스킵)
  skey_classifier.pkl, skey_feature_cols.pkl,
  skey_categorical_features.pkl, int_to_skey.pkl     (없으면 stage 3 스킵)

사용:
    python predict_component_v2.py "BIM_Data/RealData" --model-dir models_v2 --out predictions.csv

    # 파일 하나만:
    python predict_component_v2.py "BIM_Data/RealData/Equipment/foo.json" --model-dir models_v2
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.skey_mapper import skey_prefix

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.feature_extractor import extract_features_from_file

# ── 신뢰도 규칙 (predict.py와 동일한 임계값) ──
CONFIDENCE_AUTO = 0.85
CONFIDENCE_REVIEW = 0.65


def _confidence_status(conf: float) -> str:
    if conf >= CONFIDENCE_AUTO:
        return "auto"
    elif conf >= CONFIDENCE_REVIEW:
        return "review"
    else:
        return "unknown"


class ComponentPredictor:
    """group_classifier + (선택) valve_type_classifier + (선택) skey_classifier
    를 로드해서 JSON 파일 하나로부터 component_group, VALVE인 경우 valve_type
    과 개별 skey까지 함께 예측하는 래퍼."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model_dir = model_dir

        # ── 1단계: component_group (필수) ──
        self.group_model = joblib.load(model_dir / "group_classifier.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        self.categorical_cols = joblib.load(model_dir / "categorical_features.pkl")
        self.category_values = joblib.load(model_dir / "category_values.pkl")
        self.int_to_group = joblib.load(model_dir / "int_to_group.pkl")

        # feature_cols 순서/구성이 학습 시와 일치해야 하므로 run_pipeline_v2.py
        # Step 1과 동일한 키워드 목록을 유지 (현재 코드베이스에서는 kw_* 피처가
        # 실제로 생성되지 않으므로 feature_cols에 kw_*가 없다면 이 루프는
        # 아무 영향도 주지 않음 - 안전한 no-op).
        self.keyword_features = [
            "VALVE", "GASKET", "ELBOW", "TEE", "FLANGE", "OLET",
            "REDUCER", "COUPLING", "CAP", "STRAINER", "SLEEVE",
            "BELLOWS", "FLEXIBLE", "UNION", "BOLT", "WELD",
            "GAUGE", "METER", "SENSOR", "THERMOMETER", "SIGHT",
            "GLASS", "INSTRUMENT", "INDICATOR",
        ]

        # ── 2단계: valve_type (선택 - 학습이 스킵됐으면 파일이 없음) ──
        self.valve_model = None
        valve_pkl = model_dir / "valve_type_classifier.pkl"
        if valve_pkl.exists():
            self.valve_model = joblib.load(valve_pkl)
            self.valve_feature_cols = joblib.load(model_dir / "valve_feature_cols.pkl")
            self.valve_categorical_cols = joblib.load(model_dir / "valve_categorical_features.pkl")
            self.int_to_vtype = joblib.load(model_dir / "int_to_vtype.pkl")

        # ── 3단계: 개별 skey (선택 - 학습이 스킵됐으면 파일이 없음) ──
        self.skey_model = None
        skey_pkl = model_dir / "skey_classifier.pkl"
        if skey_pkl.exists():
            self.skey_model = joblib.load(skey_pkl)
            self.skey_feature_cols = joblib.load(model_dir / "skey_feature_cols.pkl")
            self.skey_categorical_cols = joblib.load(model_dir / "skey_categorical_features.pkl")
            self.int_to_skey = joblib.load(model_dir / "int_to_skey.pkl")

        print(f"[PREDICT] group ✓  valve_type {'✓' if self.valve_model else '✗ (파일 없음, 스킵)'}  "
              f"skey {'✓' if self.skey_model else '✗ (파일 없음, 스킵)'}")

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

    def predict_one(self, json_path: str | Path, source_subfolder: str = "") -> dict:
        feat = extract_features_from_file(json_path)
        X = self._build_feature_row(feat)

        group_int = int(self.group_model.predict(X)[0])
        group = self.int_to_group.get(group_int, "UNKNOWN")
        group_proba = self.group_model.predict_proba(X)[0]
        # group_proba의 컬럼 순서는 model.classes_ 순서를 따르며, 학습 시
        # rare-class 제외 등으로 클래스 일부가 빠지면 classes_가 0..N-1
        # 연속 정수가 아니게 되어 group_proba[group_int] 인덱싱이 어긋날 수
        # 있다 (엉뚱한 컬럼의 확률을 읽어 항상 0에 가까운 값이 나옴).
        # group_int는 이미 predict()의 argmax 결과이므로, 그냥 확률 벡터의
        # 최댓값을 쓰면 인덱싱 정합성과 무관하게 항상 올바른 confidence.
        group_conf = float(group_proba.max())

        result = {
            "source_subfolder": source_subfolder,
            "source_file": feat.get("source_file", Path(json_path).stem),
            "meta_family_name": feat.get("meta_family_name", ""),
            "meta_type_name": feat.get("meta_type_name", ""),
            "component_group": group,
            "component_group_confidence": round(group_conf, 4),
            "valve_type": None,
            "valve_type_confidence": None,
            "skey_prefix": None,
            "skey_confidence": None,
        }

        review_reasons = []
        if group_conf < CONFIDENCE_REVIEW:
            review_reasons.append(f"group confidence low ({group_conf:.2f})")

        # 2단계: VALVE로 예측된 경우에만 세부 타입 추가 예측
        if group == "VALVE" and self.valve_model is not None:
            X_v = X[self.valve_feature_cols]
            vtype_int = int(self.valve_model.predict(X_v)[0])
            vtype_proba = self.valve_model.predict_proba(X_v)[0]
            vtype_conf = float(vtype_proba.max())  # 위 group_conf와 동일한 이유로 max() 사용
            result["valve_type"] = self.int_to_vtype.get(vtype_int, "UNKNOWN")
            result["valve_type_confidence"] = round(vtype_conf, 4)
            if vtype_conf < CONFIDENCE_REVIEW:
                review_reasons.append(f"valve_type confidence low ({vtype_conf:.2f})")

        # 3단계: VALVE로 예측된 경우에만 개별 SKEY 추가 예측
        if group == "VALVE" and self.skey_model is not None:
            # SKEY feature tracking mirrors the general feature columns layout 
            # as saved by skey_feature_cols.pkl
            X_s = X[self.skey_feature_cols]
            
            skey_int = int(self.skey_model.predict(X_s)[0])
            skey_proba = self.skey_model.predict_proba(X_s)[0]
            skey_conf = float(skey_proba.max())  
            
            # Map the integer array index directly back to the dynamic string token
            raw_skey = self.int_to_skey.get(skey_int, "UNKNOWN")
            
            # Apply the exact prefix formatter used in Step 5 of the pipeline
            result["skey_prefix"] = skey_prefix(raw_skey) if raw_skey != "UNKNOWN" else "UNKNOWN"
            result["skey_confidence"] = round(skey_conf, 4)
            
            if skey_conf < CONFIDENCE_REVIEW:
                review_reasons.append(f"skey confidence low ({skey_conf:.2f})")

        result["review_required"] = group_conf < CONFIDENCE_AUTO
        result["review_reason"] = "; ".join(review_reasons)
        return result

    def predict_many(self, json_paths: list[tuple[str | Path, str]]) -> pd.DataFrame:
        rows = []
        for p, subfolder in json_paths:
            try:
                rows.append(self.predict_one(p, source_subfolder=subfolder))
            except Exception as e:
                rows.append({"source_file": Path(p).stem, "source_subfolder": subfolder, "error": str(e)})
        return pd.DataFrame(rows)


def _collect_json_files(path: Path) -> list[tuple[Path, str]]:
    """(json_path, source_subfolder) 튜플 리스트 반환. source_subfolder는
    `path` 기준 최상위 서브폴더명 (예: 'Equipment') - 없으면 빈 문자열."""
    if path.is_file():
        return [(path, "")]

    files = sorted(path.rglob("*.json"))
    out = []
    for f in files:
        rel = f.relative_to(path)
        subfolder = rel.parts[0] if len(rel.parts) > 1 else ""
        out.append((f, subfolder))
    return out


# ── 기본값 (사용자 환경에 맞게 필요시 여기만 고치면 됨) ──
# 인자 없이 그냥 `python predict_component_v2.py` 로 실행하면 이 값들이 쓰인다.
DEFAULT_PATH  = r"C:\Users\leonk\Documents\GitHub\huen-ai-skey\BIM_Data\RealData"
DEFAULT_MODEL_DIR = "models_v2"
DEFAULT_OUT   = r"models_v2\real_data_predictions.csv"


def main():
    parser = argparse.ArgumentParser(description="component_group + valve_type + skey 3단계 추론")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH,
                         help=f"JSON 파일 또는 JSON 파일들이 있는 폴더 (하위 폴더까지 재귀 탐색). "
                              f"생략 시 기본값: {DEFAULT_PATH}")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                         help=f"학습된 모델 폴더 (기본: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                         help=f"결과 CSV 저장 경로 (기본: {DEFAULT_OUT}). 저장하지 않으려면 --out '' 전달")
    args = parser.parse_args()
    if args.out == "":
        args.out = None

    predictor = ComponentPredictor(args.model_dir)

    path = Path(args.path)
    json_files = _collect_json_files(path)
    if not json_files:
        print(f"[오류] JSON 파일을 찾을 수 없습니다: {path}")
        return

    print(f"[PREDICT] {len(json_files)}개 파일 발견 (재귀 탐색)")
    df = predictor.predict_many(json_files)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))

    print(f"\n=== 요약 ===")
    print(f"총 {len(df)}개")
    if "component_group" in df.columns:
        print(df["component_group"].value_counts(dropna=False).to_string())
    
    # ── [추가] VALVE 카테고리의 개별 SKEY 요약 섹션 ──
    if "component_group" in df.columns and "skey_prefix" in df.columns:
        # VALVE로 분류되었고, 에러가 없는 정상적인 예측 데이터만 필터링
        valve_df = df[(df["component_group"] == "VALVE") & (df["skey_prefix"].notna())]
        
        if not valve_df.empty:
            print(f"\n[VALVE SKEY 분포] (총 {len(valve_df)}개)")
            skey_counts = valve_df["skey_prefix"].value_counts(dropna=False)
            
            # 깔끔하게 터미널에 맞춰 정렬하여 출력
            for skey, count in skey_counts.items():
                print(f"  {str(skey):<16} {count:>5}개")
        else:
            print(f"\n[VALVE SKEY 분포] 예측된 VALVE 항목이 없습니다.")
    # ──────────────────────────────────────────────────

    if "review_required" in df.columns:
        n_review = df["review_required"].sum()
        print(f"\n검토 필요: {n_review}개 ({(1-(n_review/len(df)))*100:.1f}%)")

    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()