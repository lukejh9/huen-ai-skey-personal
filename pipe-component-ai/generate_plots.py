# -*- coding: utf-8 -*-
"""
generate_plots.py
-----------------
models_v2/ 결과 CSV로부터 논문용 시각화 이미지 생성.

생성 파일:
  confusion_matrix.png         - 혼동 행렬 히트맵
  feature_importance.png       - 피처 중요도 막대그래프
  fold_accuracy.png            - Fold별 정확도
  classification_report.png    - 클래스별 Precision/Recall/F1
  class_distribution.png       - 클래스 분포
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ── 한글 폰트 설정 ──────────────────────────────
import matplotlib.font_manager as fm

# Windows 한글 폰트 후보
_KOREAN_FONTS = [
    "Malgun Gothic", "맑은 고딕",
    "NanumGothic", "나눔고딕",
    "AppleGothic", "Gulim", "Dotum"
]
_selected = None
for fn in _KOREAN_FONTS:
    try:
        fm.findfont(fm.FontProperties(family=fn), fallback_to_default=False)
        _selected = fn
        break
    except Exception:
        pass

if _selected:
    rcParams['font.family'] = _selected
else:
    rcParams['font.family'] = 'DejaVu Sans'

rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_v2")
OUT_DIR   = MODEL_DIR  # 같은 폴더에 저장

os.makedirs(OUT_DIR, exist_ok=True)

# ── 색상 팔레트 ────────────────────────────────
PRIMARY   = "#2E75B6"
SECONDARY = "#ED7D31"
ACCENT    = "#70AD47"
PALETTE   = ["#2E75B6","#ED7D31","#70AD47","#FFC000","#4472C4",
             "#FF0000","#00B0F0","#7030A0","#FF6600","#00B050",
             "#C00000","#595959"]

# ════════════════════════════════════════════════
# 1. 혼동 행렬 히트맵
# ════════════════════════════════════════════════
def plot_confusion_matrix():
    cm_df = pd.read_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"), index_col=0)
    labels = cm_df.columns.tolist()
    cm = cm_df.values.astype(float)

    # 정규화 (행 기준)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (Count)",      "d"),
        (axes[1], cm_norm, "Confusion Matrix (Normalized)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = data.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = int(data[i, j]) if fmt == "d" else data[i, j]
                txt = f"{val}" if fmt == "d" else f"{val:.2f}"
                color = "white" if data[i, j] > thresh else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=7)

    plt.suptitle("Component Group Classification – Confusion Matrix\n"
                 "Stratified 5-Fold CV (v2, Expert CSV Labels)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ════════════════════════════════════════════════
# 2. 피처 중요도
# ════════════════════════════════════════════════
def plot_feature_importance():
    fi = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))
    fi = fi.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PRIMARY if fi["importance"].iloc[i] >= fi["importance"].quantile(0.75)
              else SECONDARY
              for i in range(len(fi))]
    bars = ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="white", height=0.7)

    for bar, val in zip(bars, fi["importance"]):
        ax.text(val + 30, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Feature Importance (Sum over 5 Folds)", fontsize=11)
    ax.set_title("LightGBM Feature Importance\n"
                 "Component Group Classifier – v2 (Expert CSV Labels)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axvline(fi["importance"].quantile(0.75), color="gray",
               linestyle="--", linewidth=1, alpha=0.6, label="Q3 threshold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, fi["importance"].max() * 1.15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ════════════════════════════════════════════════
# 3. Fold별 정확도
# ════════════════════════════════════════════════
def plot_fold_accuracy():
    fa = pd.read_csv(os.path.join(MODEL_DIR, "fold_accuracies.csv"))
    mean_acc = fa["accuracy"].mean()
    std_acc  = fa["accuracy"].std()

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_colors = [PRIMARY if a >= mean_acc else SECONDARY for a in fa["accuracy"]]
    bars = ax.bar(fa["fold"], fa["accuracy"], color=bar_colors,
                  edgecolor="white", width=0.6)

    for bar, val in zip(bars, fa["accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(mean_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_acc:.4f} ± {std_acc:.4f}")
    ax.fill_between([0.5, len(fa)+0.5],
                    mean_acc - std_acc, mean_acc + std_acc,
                    color="red", alpha=0.08, label="± 1 Std")

    ax.set_xticks(fa["fold"])
    ax.set_xticklabels([f"Fold {i}" for i in fa["fold"]], fontsize=10)
    ax.set_ylabel("Validation Accuracy", fontsize=11)
    ax.set_ylim(0.90, 1.00)
    ax.set_title("Stratified 5-Fold Cross-Validation Accuracy\n"
                 "Component Group Classifier – v2 (Expert CSV Labels)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fold_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ════════════════════════════════════════════════
# 4. 분류 리포트 (클래스별 P/R/F1)
# ════════════════════════════════════════════════
def plot_classification_report():
    cr = pd.read_csv(os.path.join(MODEL_DIR, "classification_report.csv"), index_col=0)
    # 집계 행 제거
    cr = cr[~cr.index.isin(["accuracy", "macro avg", "weighted avg"])]
    cr = cr.sort_values("f1-score", ascending=True)

    x = np.arange(len(cr))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(x - w, cr["precision"], w, label="Precision", color=PRIMARY,  edgecolor="white")
    ax.barh(x,     cr["recall"],    w, label="Recall",    color=SECONDARY, edgecolor="white")
    ax.barh(x + w, cr["f1-score"],  w, label="F1-Score",  color=ACCENT,    edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(cr.index, fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.10)
    ax.axvline(0.9, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Per-Class Precision / Recall / F1-Score\n"
                 "Stratified 5-Fold CV – v2 (Expert CSV Labels)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # 샘플 수 주석
    for i, (idx, row) in enumerate(cr.iterrows()):
        ax.text(1.02, i, f"n={int(row['support'])}", va="center", fontsize=8, color="gray")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "classification_report.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ════════════════════════════════════════════════
# 5. 클래스 분포
# ════════════════════════════════════════════════
def plot_class_distribution():
    cr = pd.read_csv(os.path.join(MODEL_DIR, "classification_report.csv"), index_col=0)
    cr = cr[~cr.index.isin(["accuracy", "macro avg", "weighted avg"])]
    dist = cr["support"].sort_values(ascending=False).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 막대그래프
    colors = PALETTE[:len(dist)]
    bars = axes[0].bar(dist.index, dist.values, color=colors, edgecolor="white")
    axes[0].set_xticklabels(dist.index, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Sample Count", fontsize=11)
    axes[0].set_title("Class Distribution (Sample Count)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, dist.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 5,
                     str(val), ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # 파이차트
    wedges, texts, autotexts = axes[1].pie(
        dist.values,
        labels=dist.index,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p/100*dist.sum())})",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 8},
    )
    axes[1].set_title("Class Distribution (Proportion)", fontsize=12, fontweight="bold")

    total = dist.sum()
    fig.suptitle(f"Training Sample Distribution by Component Group\n"
                 f"Total: {total:,} samples | 180 models | Expert CSV Labels",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ════════════════════════════════════════════════
# 6. 종합 성능 비교 (v1 vs v2)
# ════════════════════════════════════════════════
def plot_version_comparison():
    """v1 (LOO-CV, 단일 모델) vs v2 (5-Fold, 181 모델) 비교."""
    versions = ["v1\n(LOO-CV\n19 models)", "v2\n(5-Fold\n180 models)"]
    # v1 결과는 이전 논문 기준 추정값 (실제 있으면 교체)
    v1_acc = 0.8831  # 이전 실험값
    v2_acc = 0.9484

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(versions, [v1_acc, v2_acc],
                  color=[SECONDARY, PRIMARY],
                  edgecolor="white", width=0.45)

    for bar, val in zip(bars, [v1_acc, v2_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.4f}\n({val*100:.2f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("CV Accuracy", fontsize=12)
    ax.set_title("Pipeline Comparison: v1 vs v2\nComponent Group Classification",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # 향상 화살표
    ax.annotate("", xy=(1, v2_acc - 0.01), xytext=(0, v1_acc + 0.01),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.text(0.5, (v1_acc + v2_acc)/2,
            f" +{(v2_acc-v1_acc)*100:.2f}%p",
            ha="center", va="center", fontsize=11, color=ACCENT, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "version_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n📊 시각화 이미지 생성 시작...")
    print(f"  출력 폴더: {OUT_DIR}\n")

    plot_confusion_matrix()
    plot_feature_importance()
    plot_fold_accuracy()
    plot_classification_report()
    plot_class_distribution()
    plot_version_comparison()

    print("\n✅ 모든 이미지 생성 완료!")
    print(f"  총 6개 파일 → {OUT_DIR}")
