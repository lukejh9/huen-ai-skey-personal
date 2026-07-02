"""
predict.py
----------
학습된 LightGBM 모델을 이용한 배관 컴포넌트 배치 추론 모듈.

추론 결과:
    component_group   : 컴포넌트 그룹 (VALVE / GASKET / SLEEVE ...)
    group_confidence  : 그룹 분류 신뢰도 (0~1)
    valve_type        : 밸브 타입 (BALL / GLOBE / ... / NON_VALVE)
    type_confidence   : 밸브 타입 신뢰도 (0~1)
    review_required   : 검토 필요 여부 (bool)
    review_reason     : 검토 사유 문자열

신뢰도 규칙:
    >= 0.9  → 자동 처리 (auto)
    0.7~0.9 → 검토 필요 (review)
    < 0.7   → 미지 (unknown / low confidence)

사용 예:
    from src.predict import PipeComponentPredictor

    predictor = PipeComponentPredictor("models/")
    result = predictor.predict_file("path/to/component.json")
    results = predictor.predict_batch("path/to/json_folder/")
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_extractor import extract_features, NUMERIC_FEATURE_COLS
from src.skey_mapper import INT_TO_GROUP, INT_TO_VTYPE, VTYPE_TO_INT


# ─────────────────────────────────────────────
#  신뢰도 규칙
# ─────────────────────────────────────────────

CONFIDENCE_AUTO    = 0.90   # 자동 처리
CONFIDENCE_REVIEW  = 0.70   # 검토 권장


def _confidence_status(conf: float) -> str:
    if conf >= CONFIDENCE_AUTO:
        return "auto"
    elif conf >= CONFIDENCE_REVIEW:
        return "review"
    else:
        return "unknown"


# ─────────────────────────────────────────────
#  Predictor 클래스
# ─────────────────────────────────────────────

class PipeComponentPredictor:
    """
    LightGBM 기반 배관 컴포넌트 분류 추론기.

    Attributes
    ----------
    group_model   : LGBMClassifier  (컴포넌트 그룹 분류기)
    group_scaler  : StandardScaler
    group_features: list[str]
    type_model    : LGBMClassifier  (밸브 타입 분류기, 없을 수 있음)
    type_scaler   : StandardScaler | None
    type_features : list[str] | None
    """

    def __init__(self, model_dir: str | Path = "models"):
        model_dir = Path(model_dir)

        # 그룹 분류기 로드 (필수)
        gm_path = model_dir / "group_classifier.pkl"
        gs_path = model_dir / "group_scaler.pkl"
        gf_path = model_dir / "group_features.pkl"

        if not gm_path.exists():
            raise FileNotFoundError(
                f"그룹 분류기를 찾을 수 없습니다: {gm_path}\n"
                "먼저 src.train_group 을 실행하세요."
            )

        self.group_model    = joblib.load(gm_path)
        self.group_scaler   = joblib.load(gs_path)
        self.group_features = joblib.load(gf_path)

        # 밸브 타입 분류기 로드 (선택)
        tm_path = model_dir / "type_classifier.pkl"
        ts_path = model_dir / "type_scaler.pkl"
        tf_path = model_dir / "type_features.pkl"

        if tm_path.exists():
            self.type_model    = joblib.load(tm_path)
            self.type_scaler   = joblib.load(ts_path)
            self.type_features = joblib.load(tf_path)
            print(f"[PREDICT] 타입 분류기 로드 완료: {tm_path}")
        else:
            self.type_model    = None
            self.type_scaler   = None
            self.type_features = None
            print(f"[PREDICT] 타입 분류기 없음 (group 분류만 사용)")

        print(f"[PREDICT] 그룹 분류기 로드 완료: {gm_path}")

    # ── 단일 피처 딕셔너리 추론 ─────────────────

    def predict_features(self, features: dict) -> dict:
        """
        피처 딕셔너리 → 분류 결과 반환.

        Parameters
        ----------
        features : dict
            extract_features() 반환 딕셔너리

        Returns
        -------
        dict
            분류 결과
        """
        # ── 그룹 분류 ──
        X_group = self._build_X(features, self.group_features)
        X_group_s = self.group_scaler.transform(X_group)

        group_proba = self.group_model.predict_proba(X_group_s)[0]
        group_idx   = int(np.argmax(group_proba))
        group_conf  = float(group_proba[group_idx])
        group_name  = INT_TO_GROUP.get(group_idx, "UNKNOWN")

        # ── 밸브 타입 분류 ──
        if group_name == "VALVE" and self.type_model is not None:
            X_type = self._build_X(features, self.type_features)
            X_type_s = self.type_scaler.transform(X_type)

            type_proba = self.type_model.predict_proba(X_type_s)[0]
            type_idx   = int(np.argmax(type_proba))
            type_conf  = float(type_proba[type_idx])
            type_name  = INT_TO_VTYPE.get(type_idx, "UNKNOWN")

            # NON_VALVE 예측이 나오면 UNKNOWN 처리
            if type_name == "NON_VALVE":
                type_name = "UNKNOWN"
                type_conf = 0.0
        else:
            type_name = "NON_VALVE"
            type_conf = group_conf if group_name != "VALVE" else 0.0

        # ── 검토 필요 여부 ──
        min_conf = min(group_conf, type_conf) if group_name == "VALVE" else group_conf
        review   = min_conf < CONFIDENCE_AUTO

        reasons = []
        if group_conf < CONFIDENCE_REVIEW:
            reasons.append(f"그룹 신뢰도 낮음 ({group_conf:.2f})")
        if group_name == "VALVE" and type_conf < CONFIDENCE_REVIEW:
            reasons.append(f"타입 신뢰도 낮음 ({type_conf:.2f})")

        return {
            "component_group":  group_name,
            "group_confidence": round(group_conf, 4),
            "group_status":     _confidence_status(group_conf),
            "valve_type":       type_name,
            "type_confidence":  round(type_conf, 4),
            "type_status":      _confidence_status(type_conf),
            "review_required":  review,
            "review_reason":    "; ".join(reasons) if reasons else "",
        }

    # ── JSON 파일 추론 ──────────────────────────

    def predict_file(self, json_path: str | Path) -> dict:
        """
        JSON 파일 한 개 → 분류 결과.

        Returns
        -------
        dict
            분류 결과 + source_file 정보
        """
        json_path = Path(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        features = extract_features(data)
        result = self.predict_features(features)
        result["source_file"]      = json_path.stem
        result["meta_family_name"] = features.get("meta_family_name", "")
        result["meta_type_name"]   = features.get("meta_type_name", "")
        return result

    # ── 폴더 배치 추론 ─────────────────────────

    def predict_batch(
        self,
        json_dir: str | Path,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        JSON 폴더 전체 배치 추론.

        Parameters
        ----------
        json_dir   : JSON 파일 폴더
        output_csv : 결과 CSV 저장 경로 (None 이면 저장 안 함)

        Returns
        -------
        pd.DataFrame
            전체 추론 결과
        """
        json_dir = Path(json_dir)
        json_files = sorted(json_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"JSON 파일 없음: {json_dir}")

        print(f"[PREDICT] {len(json_files)}개 파일 배치 추론 시작")
        rows = []

        for jf in json_files:
            try:
                result = self.predict_file(jf)
                rows.append(result)
                status = "✓" if not result["review_required"] else "⚠"
                print(
                    f"  {status} {jf.name[:50]:<50} "
                    f"→ {result['component_group']:<15} {result['valve_type']:<10} "
                    f"({result['group_confidence']:.2f})"
                )
            except Exception as e:
                print(f"  ✗ {jf.name} → 오류: {e}")

        df = pd.DataFrame(rows)

        if output_csv:
            output_csv = Path(output_csv)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"\n[PREDICT] 결과 저장: {output_csv}")

        # 요약 출력
        print(f"\n=== 배치 추론 요약 ===")
        print(f"총 {len(df)}개")
        if "component_group" in df.columns:
            print(df["component_group"].value_counts().to_string())
        if "review_required" in df.columns:
            n_review = df["review_required"].sum()
            print(f"검토 필요: {n_review}개 ({n_review/len(df)*100:.1f}%)")

        return df

    # ── 내부 유틸 ──────────────────────────────

    def _build_X(self, features: dict, feature_cols: list[str]) -> np.ndarray:
        """피처 딕셔너리 → 모델 입력 행렬 (1행)"""
        row = [features.get(c, 0.0) for c in feature_cols]
        return np.array(row, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="배관 컴포넌트 추론")
    p.add_argument("--json_dir",   type=str, required=True,
                   help="JSON 폴더 경로")
    p.add_argument("--model_dir",  type=str, default="models",
                   help="모델 폴더 (기본: models/)")
    p.add_argument("--output_csv", type=str, default=None,
                   help="결과 CSV 저장 경로")
    args = p.parse_args()

    predictor = PipeComponentPredictor(args.model_dir)
    results = predictor.predict_batch(
        args.json_dir,
        output_csv=args.output_csv,
    )
    print("\n=== 전체 결과 ===")
    print(results[[
        "source_file", "component_group", "group_confidence",
        "valve_type",  "type_confidence", "review_required"
    ]].to_string(index=False))
