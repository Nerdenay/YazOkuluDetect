"""
main.py
=======
HepaRECIST-AI: 4D Longitudinal Abdominal Tümör Takibi & RECIST 1.1 Karar Destek API'si
- C# .NET 9.0 WPF İstemcisi ile tam uyumlu REST mimarisi.
- nnU-Net v2 (Label 8 Lezyon), Radiomics Güven Skoru, Hungarian Lezyon Takibi ve Deterministik RECIST 1.1 Motoru.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import SimpleITK as sitk

# Proje Modülleri
from preprocess import convert_dicom_to_nifti, resample_image, apply_windowing_for_visualization
from inference import run_segmentation_inference, ModelUnavailableError
from recist_engine import evaluate_recist
from radiomics_module import analyze_ct_and_mask_radiomics
from matching_engine import run_longitudinal_analysis
from pipeline import run_full_pipeline, run_single_timepoint_pipeline
from report_generator import generate_clinical_report

app = FastAPI(
    title="HepaRECIST-AI API Server",
    description="4D Medikal BT Segmentasyonu, Radyomik Analiz ve RECIST 1.1 Karar Destek Sistemi",
    version="1.1.0"
)

# C# WPF Desktop istemcisi için güvenli CORS yapılandırması
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── PYDANTIC VERİ MODELLERİ ──────────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    dicom_dir: str
    output_dir: str
    output_filename: str

class PredictRequest(BaseModel):
    nifti_path: str
    output_mask_path: str
    model_dir: str = "./models"
    lesion_label_id: int = 8

class ClassifyRequest(BaseModel):
    ct_nifti_path: str
    mask_nifti_path: str
    lesion_label_id: int = 8

class LongitudinalRequest(BaseModel):
    baseline_ct_path: str
    baseline_mask_path: str
    followup_ct_path: str
    followup_mask_path: str
    output_dir: str = "./output/longitudinal"

class DecisionRequest(BaseModel):
    sod_baseline: float = Field(..., ge=0, description="Tedavi başlangıç (Baseline) hedef lezyon çap toplamı (mm)")
    sod_followup: float = Field(..., ge=0, description="Takip (Follow-up) hedef lezyon çap toplamı (mm)")
    sod_nadir: Optional[float] = Field(None, description="Önceki ölçümlerdeki en küçük SOD (Nadir). Yoksa baseline alınır.")
    new_lesion: bool = Field(False, description="Radyolog tarafından ONAYLANMIŞ yeni lezyon var mı?")
    new_lesion_candidates: int = Field(0, ge=0, description="Onay bekleyen aday yeni lezyon sayısı")

class FullPipelineRequest(BaseModel):
    baseline_dicom_dir: str
    followup_dicom_dir: str
    output_dir: str = "./output"
    model_dir: str = "./models"
    patient_id: str = "Bilinmiyor"
    patient_name: str = "Bilinmiyor"
    study_date: str = ""
    use_llm: bool = False
    llm_api_key: str = ""
    sod_nadir: Optional[float] = None
    new_lesion_confirmed: bool = False

class SinglePipelineRequest(BaseModel):
    dicom_dir: str
    output_dir: str = "./output"
    model_dir: str = "./models"
    patient_id: str = "Bilinmiyor"
    patient_name: str = "Bilinmiyor"
    study_date: str = ""

class ReportRequest(BaseModel):
    pipeline_results: dict
    patient_id: str = "Bilinmiyor"
    patient_name: str = "Bilinmiyor"
    study_date: str = ""
    use_llm: bool = False
    llm_api_key: str = ""


# ── ENDPOINTS (API UÇ NOKTALARI) ─────────────────────────────────────────────

@app.get("/", tags=["Sistem"])
def read_root():
    return {
        "status": "healthy",
        "system": "HepaRECIST-AI Decision Support Backend",
        "version": "1.1.0"
    }


@app.post("/preprocess", tags=["Ön İşleme"])
def preprocess_endpoint(request: PreprocessRequest):
    """
    DICOM serisini okur, ham HU değerleriyle 1.0mm izotropik NIfTI üretir.
    Bu dosya hem nnU-Net hem de Radiomics modülünün ortak girdisidir.
    """
    if not os.path.exists(request.dicom_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DICOM klasörü bulunamadı: {request.dicom_dir}"
        )

    try:
        os.makedirs(request.output_dir, exist_ok=True)
        raw_temp_name = "raw_" + request.output_filename
        raw_temp_path = os.path.join(request.output_dir, raw_temp_name)
        final_path = os.path.join(request.output_dir, request.output_filename)

        # 1. DICOM -> NIfTI (Ham Hounsfield Unit korunur)
        convert_dicom_to_nifti(request.dicom_dir, request.output_dir, raw_temp_name)

        # 2. 1.0 mm İzotropik Resampling (Tek seferde)
        resample_image(raw_temp_path, final_path, new_spacing=(1.0, 1.0, 1.0), is_label=False)

        # Geçici ham dosyayı temizle
        if os.path.exists(raw_temp_path) and raw_temp_path != final_path:
            os.remove(raw_temp_path)

        img = sitk.ReadImage(final_path)

        return {
            "status": "success",
            "message": "Ham HU korumalı izotropik NIfTI başarıyla oluşturuldu.",
            "file_path": final_path,
            "spacing": img.GetSpacing(),
            "size": img.GetSize()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ön işleme hatası: {str(e)}"
        )


@app.post("/predict", tags=["AI Segmentasyon"])
def predict_endpoint(request: PredictRequest):
    """
    nnU-Net v2 ile 3D Multi-Organ ve Lezyon (Label 8) segmentasyonu çıkarımı yapar.
    Model yoksa 503 döner; sahte sonuç üretmez.
    """
    return _predict(request, allow_mock=None)


@app.post("/predict-mock", tags=["AI Segmentasyon"])
def predict_mock_endpoint(request: PredictRequest):
    """YALNIZCA TEST: model yoksa sahte maske üretir (is_mock=True)."""
    return _predict(request, allow_mock=True)


def _predict(request: PredictRequest, allow_mock):
    if not os.path.exists(request.nifti_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Girdi NIfTI dosyası bulunamadı: {request.nifti_path}"
        )

    try:
        result = run_segmentation_inference(
            input_nifti_path=request.nifti_path,
            output_mask_path=request.output_mask_path,
            model_dir=request.model_dir,
            lesion_label_id=request.lesion_label_id,
            allow_mock=allow_mock
        )
        diam = result.get("primary_lesion_diameter_mm", result.get("estimated_diameter_mm", 0.0))
        return {
            "status": "success",
            "message": "AI Segmentasyonu tamamlandı." if not result["is_mock"] else "AI Segmentasyon (Mock Modu) çalıştırıldı.",
            "mask_path": result["output_mask_path"],
            "detected_lesions_count": result["detected_lesions_count"],
            "total_volume_mm3": result["total_volume_mm3"],
            "primary_lesion_diameter_mm": diam,
            "estimated_diameter_mm": diam,
            "is_mock": result["is_mock"],
            "preview_image_path": result.get("preview_image_path", "")
        }
    except ModelUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI Segmentasyon hatası: {str(e)}"
        )


@app.post("/classify", tags=["Radiomics Sınıflandırma"])
def classify_endpoint(request: ClassifyRequest):
    """
    Maskedeki lezyonların (Label 8) HU yoğunluğunu ve 3D Küreselliğini (Sphericity) analiz eder.
    """
    if not os.path.exists(request.ct_nifti_path) or not os.path.exists(request.mask_nifti_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Girdi CT veya Maske dosyası bulunamadı."
        )

    try:
        results = analyze_ct_and_mask_radiomics(
            request.ct_nifti_path, 
            request.mask_nifti_path, 
            lesion_label_id=request.lesion_label_id
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Radyomik analiz hatası: {str(e)}"
        )


@app.post("/longitudinal", tags=["Longitudinal Analiz"])
def longitudinal_endpoint(request: LongitudinalRequest):
    """
    Baseline (t0) ve Follow-up (t1) BT serilerini deforme edilebilir B-Spline ile hizalar
    ve Hungarian Algoritması ile lezyon takibi yapar.
    """
    for p in [request.baseline_ct_path, request.baseline_mask_path, request.followup_ct_path, request.followup_mask_path]:
        if not os.path.exists(p):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dosya bulunamadı: {p}"
            )

    try:
        result = run_longitudinal_analysis(
            baseline_ct=request.baseline_ct_path,
            baseline_mask=request.baseline_mask_path,
            followup_ct=request.followup_ct_path,
            followup_mask=request.followup_mask_path,
            output_dir=request.output_dir
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Longitudinal analiz hatası: {str(e)}"
        )


@app.post("/recist-decision", tags=["Karar Motoru"])
def recist_decision_endpoint(request: DecisionRequest):
    """
    RECIST 1.1 karar motoru (recist_engine.evaluate_recist; pipeline ile aynı kurallar).
    new_lesion yalnızca radyolog onaylı yeni lezyon için True gönderilmelidir.
    """
    result = evaluate_recist(
        sod_baseline=request.sod_baseline,
        sod_followup=request.sod_followup,
        sod_nadir=request.sod_nadir,
        new_lesion_confirmed=request.new_lesion,
        new_lesion_candidates=request.new_lesion_candidates,
    )
    return {"status": "success", **result}


@app.post("/pipeline", tags=["Uçtan Uca Pipeline"])
def full_pipeline_endpoint(request: FullPipelineRequest):
    """
    İki zaman noktalı (t0, t1) tüm boru hattını tek komutla çalıştırır.
    """
    for p in [request.baseline_dicom_dir, request.followup_dicom_dir]:
        if not os.path.exists(p):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"DICOM dizini bulunamadı: {p}"
            )

    try:
        patient_info = {
            "patient_id": request.patient_id,
            "patient_name": request.patient_name,
            "study_date": request.study_date or datetime.now().strftime("%Y-%m-%d")
        }

        result = run_full_pipeline(
            baseline_dicom_dir=request.baseline_dicom_dir,
            followup_dicom_dir=request.followup_dicom_dir,
            output_dir=request.output_dir,
            model_dir=request.model_dir,
            patient_info=patient_info,
            use_llm=request.use_llm,
            llm_api_key=request.llm_api_key or None,
            sod_nadir=request.sod_nadir,
            new_lesion_confirmed=request.new_lesion_confirmed
        )
        return result
    except ModelUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline hatası: {str(e)}"
        )


@app.post("/pipeline-single", tags=["Uçtan Uca Pipeline"])
def single_pipeline_endpoint(request: SinglePipelineRequest):
    """
    Tek zaman noktalı (yalnızca t0 veya tekil çekim) boru hattını çalıştırır.
    """
    if not os.path.exists(request.dicom_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DICOM dizini bulunamadı: {request.dicom_dir}"
        )

    try:
        patient_info = {
            "patient_id": request.patient_id,
            "patient_name": request.patient_name,
            "study_date": request.study_date or datetime.now().strftime("%Y-%m-%d")
        }

        result = run_single_timepoint_pipeline(
            dicom_dir=request.dicom_dir,
            output_dir=request.output_dir,
            model_dir=request.model_dir,
            patient_info=patient_info
        )
        return result
    except ModelUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tekil pipeline hatası: {str(e)}"
        )


@app.post("/generate-report", tags=["Rapor Üretimi"])
def generate_report_endpoint(request: ReportRequest):
    """
    Analiz sonuçlarından Düzce Üniversitesi resmi şablonunda klinik rapor üretir.
    """
    try:
        patient_info = {
            "patient_id": request.patient_id,
            "patient_name": request.patient_name,
            "study_date": request.study_date or datetime.now().strftime("%Y-%m-%d")
        }

        result = generate_clinical_report(
            pipeline_results=request.pipeline_results,
            patient_info=patient_info,
            use_llm=request.use_llm,
            llm_api_key=request.llm_api_key or None
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rapor üretim hatası: {str(e)}"
        )
