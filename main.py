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
from inference import run_segmentation_inference
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
    sod_nadir: Optional[float] = Field(None, description="Takip süresince görülen en küçük SOD değeri (Nadir). Yoksa baseline alınır.")
    new_lesion: bool = Field(False, description="Yeni lezyon tespit edildi mi?")

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
@app.post("/predict-mock", tags=["AI Segmentasyon"])
def predict_endpoint(request: PredictRequest):
    """
    nnU-Net v2 ile 3D Multi-Organ ve Lezyon (Label 8) segmentasyonu çıkarımı yapar.
    Ağırlıklar yoksa Mock moduna geçer.
    """
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
            lesion_label_id=request.lesion_label_id
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
    Deterministik RECIST 1.1 Kural Motoru (Eisenhauer et al., 2009).
    Nadir (tedavi süresince görülen en küçük SOD) bazlı PD kontrolü içerir.
    """
    nadir_val = request.sod_nadir if (request.sod_nadir is not None and request.sod_nadir > 0) else request.sod_baseline

    if request.sod_baseline > 0:
        change_from_baseline_pct = ((request.sod_followup - request.sod_baseline) / request.sod_baseline) * 100
    else:
        change_from_baseline_pct = 100.0 if request.sod_followup > 0 else 0.0

    if nadir_val > 0:
        change_from_nadir_pct = ((request.sod_followup - nadir_val) / nadir_val) * 100
    else:
        change_from_nadir_pct = 100.0 if request.sod_followup > 0 else 0.0

    absolute_nadir_diff = request.sod_followup - nadir_val

    # 1. Kural: Yeni Lezyon Varlığı = Kesin Progresyon (PD)
    if request.new_lesion:
        decision = "PD"
        reason = "Yeni lezyon tespit edildi. Doğrudan Progresif Hastalık (PD) kararı verildi."

    # 1b. Kural: Başlangıçta hedef lezyon yokken takipte yeni lezyon çıkması = PD
    elif request.sod_baseline == 0 and request.sod_followup > 0:
        decision = "PD"
        reason = f"Başlangıçta lezyon yokken takipte yeni lezyon yükü (+{request.sod_followup:.1f} mm) tespit edildi (PD)."

    # 1c. Kural: Başlangıçta ve takipte lezyon olmaması = CR / Lezyonsuz
    elif request.sod_baseline == 0 and request.sod_followup == 0:
        decision = "CR"
        reason = "Başlangıç ve takip tetkiklerinde hedef lezyon saptanmamıştır (Tam Yanıt / Lezyonsuz)."
    
    # 2. Kural: Nadir'e göre en az %20 VE mutlak en az 5 mm artış = Progresyon (PD)
    elif change_from_nadir_pct >= 20.0 and absolute_nadir_diff >= 5.0:
        decision = "PD"
        reason = f"Nadir ({nadir_val:.1f} mm) değerine göre %{change_from_nadir_pct:.1f} ve +{absolute_nadir_diff:.1f} mm artış tespit edildi (PD)."

    # 3. Kural: Hedef lezyonların tamamen kaybolması = Tam Yanıt (CR)
    elif request.sod_followup == 0:
        decision = "CR"
        reason = "Hedeflenen tüm lezyonlar tamamen kayboldu (Tam Yanıt - CR)."

    # 4. Kural: Baseline'a göre en az %30 azalma = Kısmi Yanıt (PR)
    elif change_from_baseline_pct <= -30.0:
        decision = "PR"
        reason = f"Baseline ({request.sod_baseline:.1f} mm) değerine göre tümör yükünde %{abs(change_from_baseline_pct):.1f} azalma tespit edildi (PR)."

    # 5. Kural: Diğer durumlar = Stabil Hastalık (SD)
    else:
        decision = "SD"
        reason = f"Belirgin küçülme veya progresyon kriteri karşılanmadı. Stabil durum (SD). (Değişim: %{change_from_baseline_pct:.1f})"

    return {
        "status": "success",
        "baseline_sod": round(request.sod_baseline, 2),
        "followup_sod": round(request.sod_followup, 2),
        "nadir_sod": round(nadir_val, 2),
        "change_from_baseline_pct": round(change_from_baseline_pct, 2),
        "change_from_nadir_pct": round(change_from_nadir_pct, 2),
        "new_lesion": request.new_lesion,
        "decision": decision,
        "explanation": reason
    }


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
            llm_api_key=request.llm_api_key or None
        )
        return result
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
