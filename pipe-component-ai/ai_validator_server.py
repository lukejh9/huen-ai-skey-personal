# -*- coding: utf-8 -*-
"""
ai_validator_server.py
----------------------
AI 모델 검증 서버.

역할:
  - proof/ 폴더 정적 파일 서비스
  - GET  /api/status              → 서버 상태 확인
  - GET  /api/models              → TrainingModel 하위 폴더 목록
  - POST /api/predict/<model>     → 폴더 전체 배치 예측
  - POST /api/predict_single      → 단일 JSON 데이터 예측 (뷰어 AI 탭용)

실행:
    cd pipe-component-ai
    python ai_validator_server.py

브라우저:
    http://localhost:8765/ai_validator.html   (전용 검증 도구)
    http://localhost:8765/revit_family_viewer.html  (기존 뷰어 + AI 탭)
"""

import csv as _csv
import json
import re as _re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# ── 경로 ─────────────────────────────────────────────
_ROOT              = Path(__file__).parent
_BIM_BASE          = _ROOT.parent / "BIM모델_데이터"
TRAINING_MODEL_DIR = _BIM_BASE / "TrainingModel"
CSV_LABEL_DIR      = _BIM_BASE / "전문가_정의_csv"
MODEL_DIR          = _ROOT / "models_v2"
PROOF_DIR          = _ROOT.parent / "proof"
PORT               = 8765

# ── sys.path 등록 ─────────────────────────────────────
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 모델 로드 ─────────────────────────────────────────
MODEL_LOADED = False
group_classifier = group_scaler = feature_cols = int_to_group = None

# 컴포넌트 그룹 → 대표 SKEY 패턴
GROUP_SKEY = {
    "VALVE":          "VB**, VT**, VG**, VY**, VC**",
    "FITTING":        "EL**, TE**, RC**, BE**, CR**",
    "FLANGE":         "FL**, FLBL, FLRF**",
    "GASKET":         "GK**, GASK",
    "SLEEVE":         "SLVE**",
    "SUPPORT":        "SP**, SS**, HS**",
    "INSTRUMENT":     "ID**, IG**, SG**, FM**",
    "FLEXIBLE_JOINT": "EX**, FX**",
    "ACCESSORY":      "OB**, SB**, ST**",
    "OLET":           "WT**, SO**, TO**, LA**",
    "CAP":            "CA**, CP**",
    "COUPLING":       "CO**, CL**, UN**",
    "TRAP":           "TRAP, PTRP",
    "PIPE":           "_P**",
    "UNKNOWN":        "NC**",
}

try:
    import joblib
    import numpy as np
    from src.feature_extractor import extract_features

    group_classifier = joblib.load(MODEL_DIR / "group_classifier.pkl")
    group_scaler     = joblib.load(MODEL_DIR / "group_scaler.pkl")
    feature_cols     = joblib.load(MODEL_DIR / "feature_cols.pkl")
    int_to_group     = joblib.load(MODEL_DIR / "int_to_group.pkl")
    MODEL_LOADED     = True
    print(f"[SERVER] 모델 로드 완료 ({len(feature_cols)}개 피처)")
except Exception as e:
    print(f"[SERVER] 모델 로드 실패: {e}")


# ── 전문가 CSV 로더 ──────────────────────────────────

def _normalize_key(text: str) -> str:
    """비교용 정규화: 공백·특수문자(콜론 포함) 제거 후 대문자."""
    return _re.sub(r"[\s_\-\.\(\):]+", "", text).upper()   # ← ':' 추가


_expert_cache: dict[str, dict] = {}  # {model_name: {fat_key: skey}}
_csv_file_map: dict[str, Path] | None = None  # 모든 CSV 파일 스템 → Path 캐시


def _get_csv_file_map() -> dict[str, Path]:
    """전문가_정의_csv 폴더의 모든 CSV를 {stem: path} 로 반환 (캐싱)."""
    global _csv_file_map
    if _csv_file_map is None:
        _csv_file_map = {}
        if CSV_LABEL_DIR.exists():
            for cp in CSV_LABEL_DIR.glob("*.csv"):
                _csv_file_map[cp.stem] = cp
        print(f"[CSV] 전문가 CSV 목록: {list(_csv_file_map.keys())}")
    return _csv_file_map


def _find_csv_for_model(model_name: str) -> Path | None:
    """
    모델 폴더명 → 대응 CSV 파일 경로 탐색.

    탐색 순서:
    1) 정확 일치:  model_name == csv_stem
    2) 접두어 일치: model_name.startswith(csv_stem)  (001_xxxx → 001.csv)
    3) 역접두어:   csv_stem.startswith(model_name)
    4) 정규화 포함: normalize(model_name) 이 normalize(csv_stem) 을 포함하거나 반대
    """
    csv_map = _get_csv_file_map()

    # 1) 정확 일치
    if model_name in csv_map:
        return csv_map[model_name]

    # 2 & 3) 접두어 일치
    for stem, path in csv_map.items():
        if model_name.startswith(stem) or stem.startswith(model_name):
            return path

    # 4) 정규화 포함 일치
    norm_model = _normalize_key(model_name)
    for stem, path in csv_map.items():
        norm_stem = _normalize_key(stem)
        if norm_model and norm_stem and (norm_model.startswith(norm_stem) or norm_stem.startswith(norm_model)):
            return path

    return None


def _load_expert_csv(model_name: str) -> tuple[dict, str]:
    """
    전문가 CSV 로드 (캐싱).
    반환: ({FamilyAndTypeName: ItemKey}, 실제_csv_파일명)
    """
    if model_name in _expert_cache:
        return _expert_cache[model_name]

    csv_path = _find_csv_for_model(model_name)
    lookup: dict[str, str] = {}
    csv_name = ""

    if csv_path and csv_path.exists():
        csv_name = csv_path.name
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    fat = row.get("FamilyAndTypeName", "").strip()
                    key = row.get("ItemKey", "").strip()
                    if fat and key:
                        lookup[fat] = key
            print(f"[CSV] '{model_name}' → {csv_name} ({len(lookup)}개 항목)")
        except Exception as e:
            print(f"[CSV] 로드 실패 '{model_name}': {e}")
    else:
        print(f"[CSV] '{model_name}'에 대응하는 CSV 없음")

    _expert_cache[model_name] = (lookup, csv_name)
    return lookup, csv_name


def _find_expert_skey(family_name: str, type_name: str, lookup: dict) -> str | None:
    """패밀리명·타입명 → 전문가 SKEY 검색. 없으면 None."""
    if not lookup:
        return None

    # 1) 정확 매칭 — 다양한 구분자 형식 시도
    candidates = []
    if type_name:
        candidates += [
            f"{family_name}: {type_name}",   # "FamilyName: TypeName"
            f"{family_name}:{type_name}",    # "FamilyName:TypeName"
            f"{family_name} {type_name}",    # "FamilyName TypeName" ← 공백만
        ]
    candidates.append(family_name)

    for cand in candidates:
        if cand in lookup:
            return lookup[cand]

    # 2) 정규화 매칭 (콜론·언더스코어·공백 모두 제거 후 비교)
    norm_map = {_normalize_key(k): v for k, v in lookup.items()}
    for cand in candidates:
        nc = _normalize_key(cand)
        if nc and nc in norm_map:
            return norm_map[nc]

    # 3) 패밀리명만으로 부분 매칭 (가장 긴 키 우선)
    norm_fam = _normalize_key(family_name)
    if norm_fam:
        matches = [(k, v) for k, v in norm_map.items()
                   if norm_fam in k or k in norm_fam]
        if matches:
            # 키 길이 내림차순 → 가장 구체적인 매칭 우선
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            return matches[0][1]

    return None


def _strip_wildcard(skey: str) -> str:
    """SKEY에서 와일드카드(**) 및 후행 공백 제거 → 접두어 반환."""
    return _re.sub(r"\*+$", "", skey.strip().upper())


def _skey_match(expert_skey: str, ai_skey_pattern: str) -> bool | None:
    """
    전문가 SKEY 접두어가 AI SKEY 패턴 접두어 목록에 포함되는지 직접 비교.

    예) expert_skey="VB**", ai_skey_pattern="VB**, VT**, VG**, VY**, VC**"
        → 접두어 "VB" ∈ {"VB","VT","VG","VY","VC"} → True

    반환값:
      True  : 포함 (일치)
      False : 미포함 (불일치)
      None  : NC** / 판단 불가
    """
    if not expert_skey:
        return None

    su = expert_skey.strip().upper()
    if su.startswith("NC") or su.startswith("_P"):
        return None          # 미분류·파이프 → 비교 제외

    expert_pfx = _strip_wildcard(su)
    if not expert_pfx:
        return None

    if not ai_skey_pattern:
        return None

    for pat in ai_skey_pattern.split(","):
        pat_pfx = _strip_wildcard(pat)
        if not pat_pfx:
            continue
        # 양방향 접두어 매칭:
        #   VB == VB  or  VB.startswith(VB)  or  VB.startswith(V) (패턴이 더 짧으면)
        if expert_pfx == pat_pfx:
            return True
        if expert_pfx.startswith(pat_pfx):   # 전문가 SKEY가 패턴 접두어로 시작
            return True
        if pat_pfx.startswith(expert_pfx):   # 패턴이 전문가 SKEY 접두어로 시작 (VB** vs V**)
            return True

    return False


def _predict_features(feat: dict) -> dict:
    """피처 딕셔너리 → 예측 결과 딕셔너리."""
    import numpy as np
    X   = np.array([feat.get(c, 0.0) for c in feature_cols], dtype=float).reshape(1, -1)
    X_s = group_scaler.transform(X)
    proba = group_classifier.predict_proba(X_s)[0]

    idx   = int(np.argmax(proba))
    conf  = float(proba[idx])
    group = int_to_group.get(idx, "UNKNOWN")

    # Top-3
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"group": int_to_group.get(int(i), "?"), "prob": round(float(proba[i]), 4)}
        for i in top3_idx
    ]

    status = "auto" if conf >= 0.8 else ("review" if conf >= 0.5 else "unknown")

    return {
        "component_group":  group,
        "skey_pattern":     GROUP_SKEY.get(group, "-"),
        "confidence":       round(conf, 4),
        "confidence_pct":   round(conf * 100, 1),
        "status":           status,
        "top3":             top3,
        "family_name":      feat.get("meta_family_name", ""),
        "type_name":        feat.get("meta_type_name", ""),
        "conn_count":       int(feat.get("conn_count", 0)),
        "conn_dia_mm":      round(feat.get("conn_radius_mean", 0) * 2 * 304.8, 1),
    }


# ── HTTP 핸들러 ──────────────────────────────────────
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".css":  "text/css",
    ".json": "application/json; charset=utf-8",
    ".png":  "image/png",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 콘솔 로그 억제

    # ── GET ─────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = unquote(parsed.path)

        # ── API ──
        if path == "/api/status":
            csv_map = _get_csv_file_map()
            self._json({
                "ok":                    True,
                "model_loaded":          MODEL_LOADED,
                "training_model_dir":    str(TRAINING_MODEL_DIR),
                "training_model_exists": TRAINING_MODEL_DIR.exists(),
                "csv_label_dir":         str(CSV_LABEL_DIR),
                "csv_label_dir_exists":  CSV_LABEL_DIR.exists(),
                "expert_csv_count":      len(csv_map),
                "expert_csv_files":      list(csv_map.keys()),
                "port":                  PORT,
            })
            return

        if path == "/api/models":
            if TRAINING_MODEL_DIR.exists():
                folders = sorted(
                    d.name for d in TRAINING_MODEL_DIR.iterdir() if d.is_dir()
                )
                counts  = {
                    d: len(list((TRAINING_MODEL_DIR / d).glob("*.json")))
                    for d in folders
                }
            else:
                folders, counts = [], {}
            self._json({
                "models":   folders,
                "counts":   counts,
                "base_dir": str(TRAINING_MODEL_DIR),
            })
            return

        # ── 정적 파일 ──
        if path in ("/", ""):
            path = "/ai_validator.html"

        fpath = PROOF_DIR / path.lstrip("/")
        if fpath.exists() and fpath.is_file():
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(fpath.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    # ── POST ────────────────────────────────────
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b"{}"
        parsed = urlparse(self.path)
        path   = unquote(parsed.path)

        # 단일 예측 (뷰어 AI 탭용)
        if path == "/api/predict_single":
            if not MODEL_LOADED:
                self._json({"error": "모델이 로드되지 않았습니다."}, 503)
                return
            try:
                from src.feature_extractor import extract_features
                data   = json.loads(body)
                feat   = extract_features(data)
                result = _predict_features(feat)
                self._json(result)
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        # 폴더 배치 예측
        if path.startswith("/api/predict/"):
            model_name = path[len("/api/predict/"):]
            model_dir  = TRAINING_MODEL_DIR / model_name
            if not MODEL_LOADED:
                self._json({"error": "모델이 로드되지 않았습니다."}, 503)
                return
            if not model_dir.exists():
                self._json({"error": f"폴더를 찾을 수 없습니다: {model_name}"}, 404)
                return

            from src.feature_extractor import extract_features
            results  = []
            errors   = []
            # 전문가 CSV 로드 (없으면 빈 dict)
            expert_lookup, expert_csv_name = _load_expert_csv(model_name)
            has_expert = bool(expert_lookup)

            json_files = sorted(model_dir.glob("*.json"))
            for jf in json_files:
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                    feat   = extract_features(data)
                    result = _predict_features(feat)
                    result["file"] = jf.stem
                    # 전문가 SKEY 추가
                    expert_skey = _find_expert_skey(
                        result.get("family_name", ""),
                        result.get("type_name", ""),
                        expert_lookup,
                    )
                    result["expert_skey"]    = expert_skey or ""
                    result["has_expert_csv"] = has_expert
                    # 일치 판단: 전문가 SKEY 접두어 ∈ AI SKEY 패턴 접두어 목록
                    if expert_skey:
                        match = _skey_match(
                            expert_skey,
                            result.get("skey_pattern", ""),
                        )
                        result["expert_group"] = _strip_wildcard(expert_skey)  # 표시용
                        result["group_match"]  = match   # True / False / None
                    else:
                        result["expert_group"] = ""
                        result["group_match"]  = None
                    results.append(result)
                except Exception as e:
                    errors.append({"file": jf.stem, "error": str(e)})

            # 통계
            from collections import Counter
            grp_cnt = Counter(r["component_group"] for r in results)
            sta_cnt = Counter(r["status"] for r in results)

            self._json({
                "model":          model_name,
                "expert_csv":     expert_csv_name,     # 실제 사용된 CSV 파일명
                "has_expert_csv": has_expert,
                "total":          len(json_files),
                "success":        len(results),
                "errors":         errors,
                "results":        results,
                "group_summary":  dict(grp_cnt),
                "status_summary": dict(sta_cnt),
            })
            return

        self.send_response(404)
        self._cors()
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── 헬퍼 ────────────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)


# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    url    = f"http://localhost:{PORT}/ai_validator.html"
    print("=" * 60)
    print("  AI Pipe Component Validation Server")
    print("=" * 60)
    print(f"  URL            : {url}")
    ok = lambda p: "OK" if p.exists() else "NOT FOUND"
    print(f"  TrainingModel  : {TRAINING_MODEL_DIR}  [{ok(TRAINING_MODEL_DIR)}]")
    print(f"  CSV_LABEL_DIR  : {CSV_LABEL_DIR}  [{ok(CSV_LABEL_DIR)}]")
    print(f"  MODEL_DIR      : {MODEL_DIR}  [{ok(MODEL_DIR)}]")
    print(f"  Model status   : {'Loaded OK' if MODEL_LOADED else 'LOAD FAILED'}")
    # CSV 파일 목록 미리 로드
    csv_map = _get_csv_file_map()
    print(f"  Expert CSVs    : {len(csv_map)} files -> {list(csv_map.keys())}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    def _open():
        import time; time.sleep(0.6)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] 종료")
