"""
api/main.py
-----------
FastAPI 기반 배관 컴포넌트 AI 분류 추론 서버.

Endpoints:
    GET  /health         : 서버 상태 확인
    POST /predict        : 피처 벡터 → 분류 결과 반환
    POST /predict/file   : JSON 파일 업로드 → 분류 결과 반환
    POST /predict/batch  : 여러 JSON 파일 업로드 → 배치 결과 반환
    GET  /model/info     : 로드된 모델 정보 반환

실행:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

예시 요청:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d @request_body.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 프로젝트 루트를 sys.path 에 추가
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.feature_extractor import NUMERIC_FEATURE_COLS, extract_features
from src.predict import PipeComponentPredictor


# ─────────────────────────────────────────────
#  앱 설정
# ─────────────────────────────────────────────

_MODEL_DIR = os.environ.get("MODEL_DIR", str(_ROOT / "models"))

app = FastAPI(
    title="Pipe Component AI Classifier",
    description=(
        "BIM Revit 패밀리 형상/커넥터/속성 피처 기반 "
        "배관 컴포넌트 자동 분류 API (LightGBM)"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 Predictor (startup 시 로드)
_predictor: PipeComponentPredictor | None = None


@app.on_event("startup")
async def startup_event():
    global _predictor
    try:
        _predictor = PipeComponentPredictor(_MODEL_DIR)
        print(f"[API] 모델 로드 완료: {_MODEL_DIR}")
    except Exception as e:
        print(f"[API] 모델 로드 실패: {e}")
        _predictor = None


# ─────────────────────────────────────────────
#  Pydantic 모델
# ─────────────────────────────────────────────

class FeatureInput(BaseModel):
    """피처 벡터 입력 스키마"""
    # 메타데이터 수치
    meta_max_size_mm: float = Field(0.0, description="최대 구경 (mm)")
    meta_min_size_mm: float = Field(0.0, description="최소 구경 (mm)")
    # 형상
    geo_surface_area: float = Field(0.0, description="표면적 (m²)")
    geo_volume:       float = Field(0.0, description="체적 (m³)")
    geo_solid_count:  float = Field(1.0, description="솔리드 수")
    geo_face_count:   float = Field(0.0, description="Face 수")
    geo_bbox_x:       float = Field(0.0, description="BBox X 크기 (m)")
    geo_bbox_y:       float = Field(0.0, description="BBox Y 크기 (m)")
    geo_bbox_z:       float = Field(0.0, description="BBox Z 크기 (m)")
    geo_bbox_diag:    float = Field(0.0, description="BBox 대각선 (m)")
    geo_aspect_xy:    float = Field(1.0, description="종횡비 X/Y")
    geo_aspect_xz:    float = Field(1.0, description="종횡비 X/Z")
    geo_aspect_yz:    float = Field(1.0, description="종횡비 Y/Z")
    geo_compactness:  float = Field(0.0, description="형상 밀집도")
    # 커넥터
    conn_count:        float = Field(0.0, description="커넥터 수")
    conn_radius_mean:  float = Field(0.0, description="커넥터 반경 평균 (m)")
    conn_radius_std:   float = Field(0.0, description="커넥터 반경 표준편차 (m)")
    conn_radius_max:   float = Field(0.0, description="커넥터 반경 최대 (m)")
    conn_radius_min:   float = Field(0.0, description="커넥터 반경 최소 (m)")
    conn_radius_range: float = Field(0.0, description="커넥터 반경 범위 (m)")
    conn_distance:     float = Field(0.0, description="커넥터 간 거리 (m)")
    conn_angle_deg:    float = Field(0.0, description="커넥터 방향 사잇각 (도)")
    conn_same_axis:    float = Field(0.0, description="동축 여부 (0/1)")
    conn_is_inline:    float = Field(0.0, description="인라인 여부 (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "meta_max_size_mm": 40.0,
                "meta_min_size_mm": 40.0,
                "geo_surface_area": 0.0186,
                "geo_volume":       0.000166,
                "geo_solid_count":  1,
                "geo_face_count":   42,
                "geo_bbox_x":       0.12,
                "geo_bbox_y":       0.12,
                "geo_bbox_z":       0.11,
                "geo_bbox_diag":    0.19,
                "geo_aspect_xy":    1.0,
                "geo_aspect_xz":    1.1,
                "geo_aspect_yz":    1.1,
                "geo_compactness":  2.1e-5,
                "conn_count":       2,
                "conn_radius_mean": 0.02,
                "conn_radius_std":  0.0,
                "conn_radius_max":  0.02,
                "conn_radius_min":  0.02,
                "conn_radius_range":0.0,
                "conn_distance":    0.037,
                "conn_angle_deg":   180.0,
                "conn_same_axis":   1,
                "conn_is_inline":   1,
            }
        }


class PredictionResult(BaseModel):
    """추론 결과 스키마"""
    component_group:  str   = Field(..., description="컴포넌트 그룹")
    group_confidence: float = Field(..., description="그룹 분류 신뢰도")
    group_status:     str   = Field(..., description="auto / review / unknown")
    valve_type:       str   = Field(..., description="밸브 타입 (비밸브 → NON_VALVE)")
    type_confidence:  float = Field(..., description="타입 분류 신뢰도")
    type_status:      str   = Field(..., description="auto / review / unknown")
    review_required:  bool  = Field(..., description="검토 필요 여부")
    review_reason:    str   = Field(..., description="검토 사유")
    inference_ms:     float = Field(..., description="추론 소요 시간 (ms)")


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health", summary="서버 상태 확인")
async def health():
    return {
        "status": "ok" if _predictor else "model_not_loaded",
        "model_dir": _MODEL_DIR,
        "group_model": _predictor is not None,
        "type_model":  _predictor is not None and _predictor.type_model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="피처 벡터 → 분류 결과",
    description=(
        "수치 피처 벡터를 입력받아 컴포넌트 그룹과 밸브 타입을 분류합니다.\n\n"
        "C# Revit Add-in 에서 추출한 피처를 직접 전달할 때 사용합니다."
    ),
)
async def predict(body: FeatureInput) -> PredictionResult:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. 먼저 학습을 수행하세요.")

    features = body.model_dump()

    t0 = time.perf_counter()
    result = _predictor.predict_features(features)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResult(**result, inference_ms=round(elapsed_ms, 2))


@app.post(
    "/predict/file",
    response_model=PredictionResult,
    summary="JSON 파일 업로드 → 분류 결과",
    description="Revit BIM JSON 파일을 업로드하면 분류 결과를 반환합니다.",
)
async def predict_file(file: UploadFile = File(...)) -> PredictionResult:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="JSON 파일만 허용됩니다.")

    contents = await file.read()
    try:
        data = json.loads(contents)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {e}")

    features = extract_features(data)

    t0 = time.perf_counter()
    result = _predictor.predict_features(features)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResult(**result, inference_ms=round(elapsed_ms, 2))


@app.post(
    "/predict/batch",
    summary="여러 JSON 파일 업로드 → 배치 결과",
    description="여러 Revit BIM JSON 파일을 업로드하여 배치 추론 결과를 반환합니다.",
)
async def predict_batch(files: list[UploadFile] = File(...)):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    results = []
    errors  = []

    for file in files:
        try:
            contents = await file.read()
            data = json.loads(contents)
            features = extract_features(data)
            result = _predictor.predict_features(features)
            result["source_file"] = file.filename
            results.append(result)
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return {
        "total":   len(files),
        "success": len(results),
        "errors":  errors,
        "results": results,
    }


@app.get("/model/info", summary="모델 정보 확인")
async def model_info():
    if _predictor is None:
        return {"status": "not_loaded"}
    return {
        "status":           "loaded",
        "group_model":      type(_predictor.group_model).__name__,
        "group_features":   _predictor.group_features,
        "group_n_features": len(_predictor.group_features) if _predictor.group_features else 0,
        "type_model":       type(_predictor.type_model).__name__ if _predictor.type_model else None,
        "type_features":    _predictor.type_features,
    }


# ─────────────────────────────────────────────
#  직접 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
