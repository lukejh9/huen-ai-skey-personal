"""
run_pipeline.py
---------------
AI 기반 배관 컴포넌트 인식 모듈 – 전체 파이프라인 실행 스크립트.

단계:
  1. BIM JSON 데이터 로드 및 피처 추출
  2. 레이블 자동 생성 (파일명 기반)
  3. 컴포넌트 그룹 분류기 학습 (LOO-CV)
  4. 밸브 타입 분류기 학습 (LOO-CV)
  5. 모델 평가 리포트 생성
  6. Ablation Study (선택)
  7. 배치 추론 결과 저장 (선택)
  8. FastAPI 서버 실행 안내

사용:
    python run_pipeline.py --data_dir ../BIM모델_데이터
    python run_pipeline.py --data_dir ../BIM모델_데이터 --ablation --infer
"""

import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트를 path에 추가
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader   import load_dataset
from src.train_group   import train_group_classifier, run_ablation_study as group_ablation
from src.train_type    import train_type_classifier,  run_ablation_study as type_ablation
from src.evaluate      import generate_evaluation_report, plot_ablation_results


# ─────────────────────────────────────────────
#  메인 파이프라인
# ─────────────────────────────────────────────

def run_pipeline(
    data_dir:   str,
    model_dir:  str = "models",
    ablation:   bool = False,
    infer:      bool = False,
    output_csv: str | None = None,
):
    t_start = time.time()
    data_dir  = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  AI 기반 배관 컴포넌트 인식 모듈 – 파이프라인 시작")
    print("=" * 65)
    print(f"  데이터 경로 : {data_dir.resolve()}")
    print(f"  모델 저장   : {model_dir.resolve()}")
    print(f"  Ablation    : {ablation}")
    print(f"  추론 실행   : {infer}")
    print("=" * 65)

    # ─ Step 1: 데이터 로드 ─────────────────────
    print("\n[STEP 1] 데이터 로드 및 피처 추출")
    # xlsx / xml 경로 자동 탐색 (data_dir 내에 있으면 자동 사용)
    xlsx_path = data_dir / "학습용레이블_정답.xlsx"
    xml_path  = data_dir / "symbol.xml"
    df = load_dataset(
        data_dir  = data_dir,
        xlsx_path = xlsx_path if xlsx_path.exists() else None,
        xml_path  = xml_path  if xml_path.exists()  else None,
        verbose   = True,
    )

    # 피처 현황 요약 CSV 저장
    feat_stat_path = model_dir / "feature_statistics.csv"
    from src.feature_extractor import NUMERIC_FEATURE_COLS
    df[NUMERIC_FEATURE_COLS].describe().T.to_csv(feat_stat_path, encoding="utf-8-sig")
    print(f"[STEP 1] 피처 통계 저장: {feat_stat_path}")

    # 레이블 현황 저장
    label_stat_path = model_dir / "label_distribution.csv"
    label_stat = df.groupby(
        ["component_group", "valve_type", "label_source"]
    ).size().reset_index(name="count")
    label_stat.to_csv(label_stat_path, index=False, encoding="utf-8-sig")
    print(f"[STEP 1] 레이블 분포 저장: {label_stat_path}")

    # xlsx 매칭 결과 저장 (검토용)
    match_path = model_dir / "label_matching.csv"
    df[["source_file", "meta_family_name", "meta_type_name",
        "skey", "component_group", "valve_type", "label_source"]
    ].to_csv(match_path, index=False, encoding="utf-8-sig")
    print(f"[STEP 1] 레이블 매칭 결과 저장: {match_path}")

    # ─ Step 2: 그룹 분류기 학습 ────────────────
    print("\n[STEP 2] 컴포넌트 그룹 분류기 학습")
    group_result = train_group_classifier(df, model_dir=model_dir, save_model=True)

    # ─ Step 3: 밸브 타입 분류기 학습 ───────────
    print("\n[STEP 3] 밸브 타입 분류기 학습")
    type_result = train_type_classifier(df, model_dir=model_dir, save_model=True)

    # ─ Step 4: 평가 리포트 생성 ────────────────
    print("\n[STEP 4] 평가 리포트 생성")
    generate_evaluation_report(group_result, type_result, output_dir=model_dir)

    # ─ Step 5: Ablation Study (선택) ───────────
    if ablation:
        print("\n[STEP 5] Ablation Study 수행")

        print("\n  [Group Classifier Ablation]")
        g_ablation_df = group_ablation(df)
        g_path = model_dir / "ablation_group.csv"
        g_ablation_df.to_csv(g_path, index=False, encoding="utf-8-sig")
        plot_ablation_results(
            g_ablation_df,
            title="Component Group Classifier – Ablation Study",
            save_path=model_dir / "ablation_group.png",
        )

        print("\n  [Valve Type Classifier Ablation]")
        t_ablation_df = type_ablation(df)
        t_path = model_dir / "ablation_type.csv"
        t_ablation_df.to_csv(t_path, index=False, encoding="utf-8-sig")
        plot_ablation_results(
            t_ablation_df,
            title="Valve Type Classifier – Ablation Study",
            save_path=model_dir / "ablation_type.png",
        )
    else:
        print("\n[STEP 5] Ablation Study 생략 (--ablation 플래그로 활성화)")

    # ─ Step 6: 배치 추론 (선택) ────────────────
    if infer:
        print("\n[STEP 6] 배치 추론 실행")
        from src.predict import PipeComponentPredictor
        predictor = PipeComponentPredictor(model_dir)
        out_csv = Path(output_csv) if output_csv else model_dir / "batch_predictions.csv"
        predictor.predict_batch(data_dir, output_csv=out_csv)
    else:
        print("\n[STEP 6] 배치 추론 생략 (--infer 플래그로 활성화)")

    # ─ 완료 요약 ────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  파이프라인 완료")
    print("=" * 65)
    print(f"  소요 시간           : {elapsed:.1f}초")
    print(f"  그룹 분류 CV Acc    : {group_result['cv_accuracy']:.4f}")
    print(f"  타입 분류 CV Acc    : {type_result['cv_accuracy']:.4f}")
    print(f"  모델 저장 위치      : {model_dir.resolve()}")
    print()
    print("  FastAPI 서버 실행 방법:")
    print(f"    cd {_ROOT}")
    print(f"    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("    → http://localhost:8000/docs  (Swagger UI)")
    print("=" * 65)

    return {
        "group_accuracy": group_result["cv_accuracy"],
        "type_accuracy":  type_result["cv_accuracy"],
        "model_dir":      str(model_dir.resolve()),
    }


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="AI 기반 배관 컴포넌트 인식 모듈 – 전체 파이프라인"
    )
    p.add_argument(
        "--data_dir", type=str, required=True,
        help="BIM JSON 폴더 경로 (예: '../BIM모델_데이터')"
    )
    p.add_argument(
        "--model_dir", type=str, default="models",
        help="모델/결과 저장 폴더 (기본: models/)"
    )
    p.add_argument(
        "--ablation", action="store_true",
        help="Ablation Study 수행 (추가 시간 소요)"
    )
    p.add_argument(
        "--infer", action="store_true",
        help="학습 후 data_dir 전체 배치 추론 수행"
    )
    p.add_argument(
        "--output_csv", type=str, default=None,
        help="배치 추론 결과 CSV 저장 경로"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        data_dir   = args.data_dir,
        model_dir  = args.model_dir,
        ablation   = args.ablation,
        infer      = args.infer,
        output_csv = args.output_csv,
    )
