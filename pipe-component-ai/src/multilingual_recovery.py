"""
multilingual_recovery.py
-------------------------
Recovery pass over nc_pending_labels.csv (items the expert CSVs left as
"NC**" / unclassified). Many of these have family/type names in French,
Chinese, German, Russian, or Japanese that clearly indicate the true
component_group even though no SKEY was assigned.

This does NOT touch the training pipeline. It produces an audit CSV
(recovered_group + which keyword triggered it + a confidence tier) so a
human can spot-check before promoting any of these to real pseudo-labels.

Design choices (deliberately conservative):
  - Longest keyword wins (mirrors the "longest-prefix" pattern already used
    in skey_mapper.py) so specific terms (e.g. "球阀" = ball valve) are
    checked before generic ones (e.g. "阀" = valve).
  - Ambiguous terms that appeared in the real data and could point to two
    different groups depending on context were deliberately LEFT OUT rather
    than guessed:
      * "REDUCTEUR"/"Réducteur" - could be a pipe reducer (FITTING) or a
        pressure-reducing valve ("Réducteur de Pression" = VALVE). Sample
        data contained both flavors.
      * "SOUDER"/"SOUDE" (French "welded") - appears as a modifier on
        weld-neck FLANGES ("BRIDE ... PLATE A SOUDER"), not as a standalone
        WELD component. Using it would mislabel flanges as welds.
    Rows that only match these are left unrecovered.
  - EXCLUSION_GUARDS: some keywords are safe almost everywhere but collide
    with one specific phrase pattern. Rather than dropping the keyword
    entirely (which would lose many good matches), a guard skips just that
    one keyword for rows containing the trigger phrase, falling through to
    the next candidate keyword (or to "unrecovered" if nothing else fits).
      * "BRIDE" (FLANGE) is guarded against "SCELLER" - "PIECE A SCELLER
        ... BRIDE" is a cast-in embedment/anchor piece (a modifier phrase
        using "bride" loosely), not a standalone flange. Confirmed by a
        confusion-matrix regression: recovering these as FLANGE measurably
        increased WELD/VALVE/CAP confusion in the v2 retrain.
  - Two confidence tiers:
      "specific"  - multi-character / compound term, low collision risk
      "generic"   - short generic word (e.g. bare "VALVE"/"阀"), kept but
                    flagged for review since collision risk is higher.
"""

import pandas as pd
from pathlib import Path

IN_PATH  = Path("/mnt/user-data/uploads/nc_pending_labels.csv")
OUT_DIR  = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# keyword -> (component_group, valve_type_or_None, tier)
# Sorted by length (desc) at lookup time, not here.
# ─────────────────────────────────────────────
KEYWORD_MAP: dict[str, tuple[str, str | None, str]] = {

    # ---- FLEXIBLE_JOINT (highest priority given WELD<->FLEXIBLE_JOINT confusion) ----
    "橡胶软接头":        ("FLEXIBLE_JOINT", None, "specific"),   # rubber flexible joint
    "橡膠軟接頭":        ("FLEXIBLE_JOINT", None, "specific"),
    "金属软接":          ("FLEXIBLE_JOINT", None, "specific"),   # metal flexible connector
    "波纹管":            ("FLEXIBLE_JOINT", None, "specific"),   # bellows / corrugated pipe
    "MANCHETTE ANTIVIBRATILE": ("FLEXIBLE_JOINT", None, "specific"),
    "MANCHON ANTIVIBRATILE":   ("FLEXIBLE_JOINT", None, "specific"),
    "ANTIVIBRATILE":     ("FLEXIBLE_JOINT", None, "specific"),
    "KOMPENSATOR":       ("FLEXIBLE_JOINT", None, "specific"),   # German: expansion joint
    "COMPENSATOR":        ("FLEXIBLE_JOINT", None, "specific"),
    "SOUFFLET":          ("FLEXIBLE_JOINT", None, "specific"),   # French: bellows

    # ---- VALVE (Chinese specific types first - longest first) ----
    "球阀":  ("VALVE", "BALL",      "specific"),
    "蝶阀":  ("VALVE", "BUTTERFLY", "specific"),
    "闸阀":  ("VALVE", "GATE",      "specific"),
    "截止阀": ("VALVE", "GLOBE",    "specific"),
    "止回阀": ("VALVE", "CHECK",    "specific"),
    "减压阀": ("VALVE", "PRESSURE_REDUCING", "specific"),
    "塞閥":  ("VALVE", "PLUG",      "specific"),
    "阀":    ("VALVE", None,        "generic"),
    "閥":    ("VALVE", None,        "generic"),

    # ---- VALVE (French) ----
    "ROBINET A BOISSEAU SPHERIQUE": ("VALVE", "BALL", "specific"),
    "SOUPAPE DE SECURITE": ("VALVE", "SAFETY", "specific"),
    "ELECTROVANNE":  ("VALVE", "CONTROL", "specific"),
    "CLAPET ANTI-RETOUR": ("VALVE", "CHECK", "specific"),
    "CLAPET":        ("VALVE", "CHECK", "specific"),
    "GUILLOTINE":    ("VALVE", "KNIFE_GATE", "specific"),
    "PAPILLON":      ("VALVE", "BUTTERFLY", "specific"),
    "BOISSEAU SPHERIQUE": ("VALVE", "BALL", "specific"),
    "ROBINET":       ("VALVE", None, "generic"),
    "VANNE":         ("VALVE", None, "generic"),
    "SOUPAPE":       ("VALVE", None, "generic"),

    # ---- VALVE (German) ----
    "KUGELHAHN":     ("VALVE", "BALL", "specific"),
    "ABSPERRKLAPPE": ("VALVE", "BUTTERFLY", "specific"),

    # ---- VALVE (Russian - Cyrillic) ----
    "ШАРОВЫЙ":       ("VALVE", "BALL", "specific"),
    "ШАРОВОЙ":       ("VALVE", "BALL", "specific"),
    "ЗАДВИЖКА":      ("VALVE", "GATE", "specific"),
    "КРАН":          ("VALVE", None, "generic"),
    "КЛАПАН":        ("VALVE", None, "generic"),

    # ---- VALVE (Japanese) ----
    "チェック バルブ": ("VALVE", "CHECK", "specific"),
    "グローブ バルブ": ("VALVE", "GLOBE", "specific"),
    "バルブ":         ("VALVE", None, "generic"),

    # ---- FLANGE ----
    "法蘭":    ("FLANGE", None, "specific"),
    "法兰":    ("FLANGE", None, "specific"),
    "FLANSCH": ("FLANGE", None, "specific"),
    "BRIDE":   ("FLANGE", None, "specific"),
    "플랜지":   ("FLANGE", None, "specific"),
    "ФЛАНЕЦ":  ("FLANGE", None, "specific"),
    "フランジ": ("FLANGE", None, "specific"),

    # ---- COUPLING ----
    "管接头":   ("COUPLING", None, "specific"),
    "管接頭":   ("COUPLING", None, "specific"),
    "RACCORD": ("COUPLING", None, "specific"),
    "ХОМУТ":   ("COUPLING", None, "specific"),
    "MANCHETTE": ("COUPLING", None, "generic"),
    "MANCHON":   ("COUPLING", None, "generic"),

    # ---- INSTRUMENT ----
    "压力表":       ("INSTRUMENT", None, "specific"),
    "溫度計":       ("INSTRUMENT", None, "specific"),
    "温度计":       ("INSTRUMENT", None, "specific"),
    "流量计":       ("INSTRUMENT", None, "specific"),
    "流量計":       ("INSTRUMENT", None, "specific"),
    "DEBITMETRE":  ("INSTRUMENT", None, "specific"),
    "DÉBITMÈTRE":  ("INSTRUMENT", None, "specific"),
    "MANOMETRE":   ("INSTRUMENT", None, "specific"),
    "COMPTEUR":    ("INSTRUMENT", None, "generic"),
    "WATER_METER": ("INSTRUMENT", None, "specific"),

    # ---- ACCESSORY (filter/strainer) ----
    "过滤器":  ("ACCESSORY", None, "specific"),
    "濾網":    ("ACCESSORY", None, "specific"),
    "滤网":    ("ACCESSORY", None, "specific"),
    "FILTRE":  ("ACCESSORY", None, "generic"),

    # ---- SUPPORT ----
    "ОПОРА": ("SUPPORT", None, "specific"),
}

# sort once by keyword length, descending -> specific matches win over generic substrings
_SORTED_KEYWORDS = sorted(KEYWORD_MAP.items(), key=lambda kv: -len(kv[0]))


# ─────────────────────────────────────────────
# keyword -> [trigger phrases that veto this keyword's match]
# If any trigger phrase is present alongside the keyword, that keyword is
# skipped for this row (the loop falls through to the next candidate
# keyword instead of returning a match). Add new guards here rather than
# deleting a generally-useful keyword outright.
# ─────────────────────────────────────────────
EXCLUSION_GUARDS: dict[str, list[str]] = {
    # "PIECE A SCELLER ... BRIDE" = a cast-in embedment/anchor piece, not a
    # standalone flange. Found via confusion-matrix regression after the
    # first recovery pass (see docstring above).
    "BRIDE": ["SCELLER"],
}


# Tier ranking used to enforce a `min_tier` cutoff. Higher = stricter/safer.
_TIER_RANK = {"generic": 0, "specific": 1}


def recover_row(
    family: str,
    type_: str,
    min_tier: str = "specific",
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Look up (family, type) text against the multilingual keyword map.

    Parameters
    ----------
    min_tier : "specific" (default, conservative) or "generic".
        Rows that only match a "generic" keyword (e.g. bare "VALVE"/"阀")
        are ignored when min_tier="specific". Pass min_tier="generic" to
        include those too (only recommended after manual spot-checking -
        see nc_recovered_labels.csv's "generic" tier rows).

    Returns
    -------
    (group, valve_type, matched_keyword, tier) or (None, None, None, None)
    if nothing matched at or above min_tier.
    """
    if min_tier not in _TIER_RANK:
        raise ValueError(f"min_tier must be one of {list(_TIER_RANK)}, got {min_tier!r}")
    min_rank = _TIER_RANK[min_tier]

    text = f"{family or ''} {type_ or ''}".upper()
    for kw, (group, vtype, tier) in _SORTED_KEYWORDS:
        if _TIER_RANK[tier] < min_rank:
            continue
        if kw not in text:
            continue
        guards = EXCLUSION_GUARDS.get(kw)
        if guards and any(trigger in text for trigger in guards):
            continue
        return group, vtype, kw, tier
    return None, None, None, None


def main():
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

    # Standalone audit run: include both tiers, tier column lets you filter.
    results = df.apply(
        lambda r: recover_row(r.get("family", ""), r.get("type", ""), min_tier="generic"),
        axis=1, result_type="expand",
    )
    results.columns = ["recovered_group", "recovered_valve_type", "matched_keyword", "confidence_tier"]
    out = pd.concat([df, results], axis=1)

    recovered = out[out["recovered_group"].notna()].copy()
    unrecovered = out[out["recovered_group"].isna()].copy()

    recovered.to_csv(OUT_DIR / "nc_recovered_labels.csv", index=False, encoding="utf-8-sig")
    unrecovered.drop(columns=["recovered_group", "recovered_valve_type", "matched_keyword", "confidence_tier"]) \
               .to_csv(OUT_DIR / "nc_still_unrecovered.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("  Multilingual NC** Recovery Pass")
    print("=" * 60)
    print(f"  Total NC** rows          : {len(out)}")
    print(f"  Recovered                : {len(recovered)} ({len(recovered)/len(out)*100:.1f}%)")
    print(f"  Still unrecovered        : {len(unrecovered)}")
    print()
    print("  Recovered by group:")
    print(recovered["recovered_group"].value_counts().to_string())
    print()
    print("  Recovered by confidence tier:")
    print(recovered["confidence_tier"].value_counts().to_string())
    print()
    print("  -> FLEXIBLE_JOINT / WELD recovered (directly relevant to the")
    print("     WELD<->FLEXIBLE_JOINT confusion found in the confusion matrix):")
    fj_weld = recovered[recovered["recovered_group"].isin(["FLEXIBLE_JOINT", "WELD"])]
    print(f"     {len(fj_weld)} rows")
    if len(fj_weld):
        print(fj_weld[["model", "family", "type", "recovered_group", "matched_keyword"]].to_string(index=False))
    print()
    print(f"  Saved: {OUT_DIR / 'nc_recovered_labels.csv'}")
    print(f"  Saved: {OUT_DIR / 'nc_still_unrecovered.csv'}")

    # "generic" tier rows recovered via a single common word deserve a closer look
    generic = recovered[recovered["confidence_tier"] == "generic"]
    print(f"\n  [REVIEW RECOMMENDED] {len(generic)} rows matched only on a generic term")
    print(f"  (e.g. bare 'VALVE'/'阀'/'FILTRE') - spot-check before using as labels.")


if __name__ == "__main__":
    main()