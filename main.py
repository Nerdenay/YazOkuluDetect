import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Preprocess, Inference ve Radiomics modüllerimizden fonksiyonları import ediyoruz
from preprocess import convert_dicom_to_nifti, apply_windowing, resample_image
from inference import run_segmentation_inference
from radiomics_module import analyze_ct_and_mask_radiomics
from matching_engine import run_longitudinal_analysis
from pipeline import run_full_pipeline, run_single_timepoint_pipeline
from report_generator import generate_clinical_report

app = FastAPI(
    title="RECIST 1.1 Decision Support System API",
    description="Medikal BT analizi, lezyon tespiti, Radiomics sınıflandırma ve RECIST 1.1 kural motoru için FastAPI backend sunucusu.",
    version="1.0.0"
)

# C# WPF (localhost) ile çapraz köken HTTP haberleşmesine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC VERİ MODELLERİ (İstek Yapılarını Doğrulamak İçin) ---

class PreprocessRequest(BaseModel):
    dicom_dir: str
    output_dir: str
    output_filename: str

class PredictRequest(BaseModel):
    nifti_path: str
    output_mask_path: str
    model_dir: str = "./models"

class ClassifyRequest(BaseModel):
    ct_nifti_path: str
    mask_nifti_path: str

class LongitudinalRequest(BaseModel):
    baseline_ct_path: str
    baseline_mask_path: str
    followup_ct_path: str
    followup_mask_path: str
    output_dir: str = "./output/longitudinal"

class DecisionRequest(BaseModel):
    sod_baseline: float
    sod_followup: float
    new_lesion: bool

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

# --- ENDPOINTS (API UÇ NOKTALARI) ---

@app.get("/", tags=["Sistem"])
def read_root():
    """
    Sunucunun çalışıp çalışmadığını kontrol etmek için Health-Check uç noktası.
    """
    return {
        "status": "healthy",
        "message": "RECIST 1.1 Karar Destek Sistemi API Sunucusu Aktif!"
    }

@app.post("/preprocess", tags=["Ön İşleme"])
def preprocess_endpoint(request: PreprocessRequest):
    """
    DICOM serisini okur, NIfTI formatına çevirir, HU pencerelemesi (WL:40, WW:150)
    uygular ve izotropik (1.0mm) çözünürlüğe yeniden örnekler (resample).
    
    ÖNEMLİ: İki çıktı üretir:
      1) Normalize edilmiş NIfTI (nnU-Net segmentasyon girdisi)  -> output_filename
      2) Ham HU NIfTI (Radiomics sınıflandırma girdisi)          -> raw_hu_<output_filename>
    
    Radiomics modülü HU eşikleriyle çalıştığı için (kist: 0-20 HU, tümör: 30-85 HU),
    normalize [0,1] aralığındaki veriyle değil, ham HU değerleriyle beslenmelidir.
    """
    if not os.path.exists(request.dicom_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Belirtilen DICOM klasörü bulunamadı: {request.dicom_dir}"
        )
        
    try:
        raw_name = "raw_" + request.output_filename
        windowed_name = "windowed_" + request.output_filename
        raw_hu_name = "raw_hu_" + request.output_filename
        
        # 1. Adım: DICOM -> NIfTI (Ham HU değerleri)
        raw_path = convert_dicom_to_nifti(request.dicom_dir, request.output_dir, raw_name)
        
        # 2. Adım: Resampling (1.0 mm isotropic) — Ham HU kopyasını Radiomics için sakla
        raw_hu_resampled_path = os.path.join(request.output_dir, raw_hu_name)
        resample_image(raw_path, raw_hu_resampled_path, new_spacing=(1.0, 1.0, 1.0), is_label=False)
        
        # 3. Adım: HU Windowing [-35, 115] HU kırpma ve [0,1] normalizasyon
        windowed_path = os.path.join(request.output_dir, windowed_name)
        apply_windowing(raw_path, windowed_path)
        
        # 4. Adım: Resampling (1.0 mm isotropic) — Normalize edilmiş versiyon (nnU-Net girdisi)
        final_path = os.path.join(request.output_dir, request.output_filename)
        resample_image(windowed_path, final_path, new_spacing=(1.0, 1.0, 1.0), is_label=False)
        
        # Geçici ara dosyaları temizle (ham dönüşüm ve pencerelenmiş ara dosyaları sil)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(windowed_path):
            os.remove(windowed_path)
            
        img = sitk.ReadImage(final_path)
        
        return {
            "status": "success",
            "message": "Ön işleme adımları başarıyla tamamlandı.",
            "file_path": final_path,
            "raw_hu_path": raw_hu_resampled_path,
            "spacing": img.GetSpacing(),
            "size": img.GetSize()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ön işleme sırasında sunucu hatası: {str(e)}"
        )

@app.post("/predict", tags=["AI Segmentasyon"])
@app.post("/predict-mock", tags=["AI Segmentasyon"])
def predict_endpoint(request: PredictRequest):
    """
    nnU-Net 3D Lezyon Segmentasyonu Çıkarım Motoru.
    Eğitilmiş model ağırlıkları varsa gerçek AI tahmini yapar, aksi halde güvenli Mock modunu kullanır.
    """
    if not os.path.exists(request.nifti_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Giriş NIfTI dosyası bulunamadı: {request.nifti_path}"
        )
        
    try:
        result = run_segmentation_inference(
            input_nifti_path=request.nifti_path,
            output_mask_path=request.output_mask_path,
            model_dir=request.model_dir
        )
        
        return {
            "status": "success",
            "message": "AI Segmentasyon tahmini tamamlandı." if not result["is_mock"] else "AI Segmentasyon (Mock Modu) çalıştırıldı.",
            "mask_path": result["output_mask_path"],
            "detected_lesions_count": result["detected_lesions_count"],
            "total_volume_mm3": result["total_volume_mm3"],
            "estimated_diameter_mm": result["estimated_diameter_mm"],
            "is_mock": result["is_mock"],
            "preview_image_path": result.get("preview_image_path", "")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI Segmentasyon tahmini sırasında hata: {str(e)}"
        )

@app.post("/classify", tags=["Radiomics Sınıflandırma"])
def classify_endpoint(request: ClassifyRequest):
    """
    Radyomik Özellik Çıkarımı ve Sınıflandırma Endpoint'i.
    Segmentasyonu yapılmış 3D lezyon adaylarını Malign / Benign (Kist) / Vasküler 
    olarak ayırır ve Güven Skoru üretir.
    """
    if not os.path.exists(request.ct_nifti_path) or not os.path.exists(request.mask_nifti_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Girdi CT veya Maske NIfTI dosyaları bulunamadı."
        )
        
    try:
        results = analyze_ct_and_mask_radiomics(request.ct_nifti_path, request.mask_nifti_path)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Radyomik analiz sırasında hata oluştu: {str(e)}"
        )

@app.post("/longitudinal", tags=["Longitudinal Analiz"])
def longitudinal_endpoint(request: LongitudinalRequest):
    """
    Longitudinal Lezyon Takip ve Eşleştirme Boru Hattı.
    Baseline (t0) ve Follow-up (t1) CT + maske çiftlerini alır, registration yapar,
    Hungarian Algorithm ile lezyonları eşleştirir ve RECIST 1.1 girdilerini otomatik hesaplar.
    """
    for path in [request.baseline_ct_path, request.baseline_mask_path, 
                 request.followup_ct_path, request.followup_mask_path]:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dosya bulunamadı: {path}"
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
            detail=f"Longitudinal analiz sırasında hata: {str(e)}"
        )

@app.post("/recist-decision", tags=["Karar Motoru"])
def recist_decision_endpoint(request: DecisionRequest):
    """
    RECIST 1.1 Kural Motoru. 
    Baz ve Kontrol tümör çap toplamlarına (SOD) ve yeni lezyon durumuna göre
    CR, PR, SD veya PD kararlarını tamamen kurallara bağlı olarak üretir.
    """
    if request.sod_baseline <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Baz SOD değeri sıfırdan büyük olmalıdır."
        )
        
    change_pct = ((request.sod_followup - request.sod_baseline) / request.sod_baseline) * 100
    
    # RECIST 1.1 Standart Kuralları (Eisenhauer et al., 2009)
    if request.new_lesion:
        decision = "PD"
        reason = "Yeni lezyon tespit edildi. (iRECIST gereği teyit gerekebilir)"
    elif change_pct >= 20.0 and (request.sod_followup - request.sod_baseline) >= 5.0:
        decision = "PD"
        reason = f"Tümör yükünde en az %20 ve mutlak olarak en az 5mm artış tespit edildi (Değişim: %{change_pct:.1f})."
    elif request.sod_followup == 0:
        decision = "CR"
        reason = "Hedeflenen tüm lezyonlar tamamen kayboldu (Tam Yanıt)."
    elif change_pct <= -30.0:
        decision = "PR"
        reason = f"Tümör yükünde en az %30 azalma tespit edildi (Değişim: %{change_pct:.1f})."
    else:
        decision = "SD"
        reason = f"Belirgin bir küçülme veya büyüme tespit edilmedi. Stabil durum (Değişim: %{change_pct:.1f})."
        
    return {
        "status": "success",
        "baseline_sod": request.sod_baseline,
        "followup_sod": request.sod_followup,
        "change_percentage": round(change_pct, 2),
        "new_lesion": request.new_lesion,
        "decision": decision,
        "explanation": reason
    }

@app.post("/pipeline", tags=["Uçtan Uca Pipeline"])
def full_pipeline_endpoint(request: FullPipelineRequest):
    """
    Uçtan Uca RECIST 1.1 Analiz Boru Hattı.
    İki zamanlı (baseline + follow-up) DICOM setini alır ve tek çağrıda:
    Preprocessing → Segmentasyon → Radiomics → Registration → RECIST → Rapor üretir.
    """
    for path in [request.baseline_dicom_dir, request.followup_dicom_dir]:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"DICOM klasörü bulunamadı: {path}"
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
            llm_api_key=request.llm_api_key if request.llm_api_key else None
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
    Tek Zamanlı Analiz Boru Hattı.
    İlk çekim veya tek CT seti için: Preprocessing → Segmentasyon → Radiomics.
    RECIST kararı üretilemez (karşılaştırma yok).
    """
    if not os.path.exists(request.dicom_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DICOM klasörü bulunamadı: {request.dicom_dir}"
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
            detail=f"Pipeline hatası: {str(e)}"
        )

@app.post("/generate-report", tags=["Rapor Üretimi"])
def generate_report_endpoint(request: ReportRequest):
    """
    Mevcut pipeline sonuçlarından klinik rapor üretir.
    Şablon tabanlı veya LLM (Gemini API) destekli rapor oluşturur.
    LLM tıbbi karar VERMEZ — sadece doğrulanmış verileri rapor diline çevirir.
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
            llm_api_key=request.llm_api_key if request.llm_api_key else None
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rapor üretim hatası: {str(e)}"
        )
