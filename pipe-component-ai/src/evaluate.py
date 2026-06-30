"""
evaluate.py
-----------
모델 평가 유틸리티.

기능:
  - Feature Importance 시각화
  - Confusion Matrix 시각화
  - Ablation Study 결과 시각화
  - 분류 보고서 출력
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경 대응
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 한글 폰트 설정 (Windows 환경)
_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
for _f in _FONTS:
    if any(_f in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
#  Feature Importance 시각화
# ─────────────────────────────────────────────

def plot_feature_importance(
    fi_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: str | Path | None = None,
):
    """
    Feature Importance 막대 그래프.

    Parameters
    ----------
    fi_df : pd.DataFrame
        컬럼: ["feature", "importance"]
    title : str
    top_n : int
        상위 N개만 표시
    save_path : str | Path | None
        저장 경로 (None 이면 plt.show())
    """
    top = fi_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    bars = ax.barh(top["feature"], top["importance"], color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (split count)")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[EVAL] Feature Importance 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  Confusion Matrix 시각화
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: str | None = "true",
):
    """
    Confusion Matrix 히트맵.

    Parameters
    ----------
    y_true      : 실제 레이블 (정수)
    y_pred      : 예측 레이블 (정수)
    class_names : 클래스 이름 목록
    title       : 그래프 제목
    save_path   : 저장 경로
    normalize   : "true" (행 정규화) | "pred" | "all" | None
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    labels = class_names

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues",
              values_format=".2f" if normalize else "d")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[EVAL] Confusion Matrix 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  Ablation Study 결과 시각화
# ─────────────────────────────────────────────

def plot_ablation_results(
    ablation_df: pd.DataFrame,
    title: str = "Ablation Study",
    save_path: str | Path | None = None,
):
    """
    Ablation Study 결과 막대 그래프.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        컬럼: ["feature_set", "cv_accuracy"]
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(ablation_df))
    bars = ax.bar(
        x,
        ablation_df["cv_accuracy"] * 100,
        color="steelblue",
        edgecolor="white",
        width=0.6,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ablation_df["feature_set"], rotation=20, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("CV Accuracy (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[EVAL] Ablation 결과 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  전체 평가 리포트 생성
# ─────────────────────────────────────────────

def generate_evaluation_report(
    group_result: dict,
    type_result: dict,
    output_dir: str | Path = "models",
):
    """
    그룹 / 타입 분류기의 평가 결과를 시각화 파일로 저장.

    Parameters
    ----------
    group_result : train_group_classifier() 반환값
    type_result  : train_type_classifier() 반환값
    output_dir   : 저장 폴더
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 그룹 분류기 Feature Importance ──
    if group_result.get("feature_importance") is not None:
        plot_feature_importance(
            group_result["feature_importance"],
            title="Component Group Classifier – Feature Importance",
            save_path=output_dir / "group_feature_importance.png",
        )

    # ── 타입 분류기 Feature Importance ──
    if type_result.get("feature_importance") is not None:
        plot_feature_importance(
            type_result["feature_importance"],
            title="Valve Type Classifier – Feature Importance",
            save_path=output_dir / "type_feature_importance.png",
        )

    # ── 텍스트 보고서 저장 ──
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  BIM 배관 컴포넌트 AI 분류기 평가 보고서\n")
        f.write("=" * 60 + "\n\n")

        f.write("[컴포넌트 그룹 분류기]\n")
        f.write(f"  LOO-CV Accuracy : {group_result.get('cv_accuracy', 'N/A'):.4f}\n\n")
        f.write(group_result.get("report", "") + "\n\n")

        f.write("[밸브 타입 분류기]\n")
        f.write(f"  LOO-CV Accuracy : {type_result.get('cv_accuracy', 'N/A'):.4f}\n\n")
        f.write(type_result.get("report", "") + "\n")

    print(f"[EVAL] 평가 보고서 저장: {report_path}")


if __name__ == "__main__":
    print("evaluate.py: 시각화 유틸리티 모듈 (직접 실행 불가, run_pipeline.py 를 이용하세요)")
