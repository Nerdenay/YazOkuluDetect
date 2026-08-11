import os
from typing import Dict, Any, Optional
from datetime import datetime

# Tüm alt modülleri import et
from preprocess import convert_dicom_to_nifti, apply_windowing, resample_image
from inference import run_segmentation_inference
from radiomics_module import analyze_ct_and_mask_radiomics
from matching_engine import run_longitudinal_analysis
from report_generator import generate_clinical_report


def run_full_pipeline(baseline_dicom_dir: str, followup_dicom_dir: str,
                       output_dir: str = "./output",
                       model_dir: str = "./models",
                       patient_info: Optional[Dict[str, str]] = None,
                       use_llm: bool = False,
                       llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    RECIST 1.1 Karar Destek Sistemi — Uçtan Uca Boru Hattı Orkestrasyonu.
    
    Tek bir çağrıyla tüm analiz sürecini başlatır ve yönetir:
    
    DICOM (t0 + t1) 
        → Preprocessing (HU Windowing + Resampling)
        → AI Segmentasyon (nnU-Net / Mock)
        → Radiomics Sınıflandırma (Malign / Benign / Vasküler)
        → Longitudinal Registration + Hungarian Eşleştirme
        → RECIST 1.1 Deterministik Karar Motoru
        → Klinik Rapor Üretimi (Şablon veya LLM)
    
    Args:
        baseline_dicom_dir: Baseline (t0) DICOM klasörü
        followup_dicom_dir: Follow-up (t1) DICOM klasörü
        output_dir: Tüm çıktıların kaydedileceği ana klasör
        model_dir: nnU-Net model ağırlıkları klasörü
        patient_info: Hasta bilgileri dict'i (patient_id, patient_name, study_date)
        use_llm: LLM rapor modu aktif mi?
        llm_api_key: Gemini API anahtarı
        
    Returns:
        Tüm aşamaların sonuçlarını içeren kapsamlı dict
    """
    print("=" * 70)
    print("  RECIST 1.1 KARAR DESTEK SİSTEMİ — UÇTAN UCA BOR HATTI")
    print("=" * 70)
    start_time = datetime.now()
    
    # Çıktı alt klasörlerini hazırla
    preprocess_dir = os.path.join(output_dir, "preprocessed")
    masks_dir = os.path.join(output_dir, "masks")
    longitudinal_dir = os.path.join(output_dir, "longitudinal")
    reports_dir = os.path.join(output_dir, "reports")
    
    for d in [preprocess_dir, masks_dir, longitudinal_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
    
    results = {
        "pipeline_version": "1.0.0",
        "started_at": start_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # =====================================================================
    # ADIM 1: PREPROCESSING — DICOM → NIfTI + HU Windowing + Resampling
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 1/6] Preprocessing: DICOM → NIfTI dönüşümü ve normalizasyon...")
    print(f"{'─' * 70}")
    
    # Baseline (t0) preprocessing
    print("\n  [t0] Baseline preprocessing...")
    bl_raw = convert_dicom_to_nifti(baseline_dicom_dir, preprocess_dir, "raw_baseline.nii.gz")
    bl_raw_hu = os.path.join(preprocess_dir, "raw_hu_baseline.nii.gz")
    resample_image(bl_raw, bl_raw_hu, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    bl_windowed = os.path.join(preprocess_dir, "windowed_baseline.nii.gz")
    apply_windowing(bl_raw, bl_windowed)
    bl_final = os.path.join(preprocess_dir, "baseline.nii.gz")
    resample_image(bl_windowed, bl_final, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    # Geçici dosyaları temizle
    for tmp in [bl_raw, bl_windowed]:
        if os.path.exists(tmp):
            os.remove(tmp)
    
    # Follow-up (t1) preprocessing
    print("  [t1] Follow-up preprocessing...")
    fu_raw = convert_dicom_to_nifti(followup_dicom_dir, preprocess_dir, "raw_followup.nii.gz")
    fu_raw_hu = os.path.join(preprocess_dir, "raw_hu_followup.nii.gz")
    resample_image(fu_raw, fu_raw_hu, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    fu_windowed = os.path.join(preprocess_dir, "windowed_followup.nii.gz")
    apply_windowing(fu_raw, fu_windowed)
    fu_final = os.path.join(preprocess_dir, "followup.nii.gz")
    resample_image(fu_windowed, fu_final, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    for tmp in [fu_raw, fu_windowed]:
        if os.path.exists(tmp):
            os.remove(tmp)
    
    results["preprocessing"] = {
        "baseline_nifti": bl_final,
        "baseline_raw_hu": bl_raw_hu,
        "followup_nifti": fu_final,
        "followup_raw_hu": fu_raw_hu
    }
    print("  [✓] Preprocessing tamamlandı.")
    
    # =====================================================================
    # ADIM 2: AI SEGMENTASYON — nnU-Net veya Mock
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 2/6] AI Segmentasyon: 3D Lezyon Tespiti...")
    print(f"{'─' * 70}")
    
    bl_mask = os.path.join(masks_dir, "baseline_mask.nii.gz")
    fu_mask = os.path.join(masks_dir, "followup_mask.nii.gz")
    
    print("  [t0] Baseline segmentasyon...")
    bl_seg = run_segmentation_inference(bl_final, bl_mask, model_dir)
    print("  [t1] Follow-up segmentasyon...")
    fu_seg = run_segmentation_inference(fu_final, fu_mask, model_dir)
    
    results["segmentation"] = {
        "baseline": bl_seg,
        "followup": fu_seg,
        "is_mock": bl_seg["is_mock"]
    }
    print(f"  [✓] Segmentasyon tamamlandı. (Mock: {bl_seg['is_mock']})")
    
    # =====================================================================
    # ADIM 3: RADYOMİK SINIFLANDIRMA — Malign / Benign / Vasküler
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 3/6] Radiomics Sınıflandırma: Komşu Patoloji Ayrımı...")
    print(f"{'─' * 70}")
    
    # Radiomics ham HU CT ile çalışır (normalize CT değil!)
    print("  [t0] Baseline radiomics analizi...")
    bl_radiomics = analyze_ct_and_mask_radiomics(bl_raw_hu, bl_mask)
    print("  [t1] Follow-up radiomics analizi...")
    fu_radiomics = analyze_ct_and_mask_radiomics(fu_raw_hu, fu_mask)
    
    results["radiomics"] = {
        "baseline": bl_radiomics,
        "followup": fu_radiomics,
        "requires_radiologist_review": (
            bl_radiomics.get("requires_radiologist_review", False) or
            fu_radiomics.get("requires_radiologist_review", False)
        )
    }
    print(f"  [✓] Radiomics sınıflandırma tamamlandı.")
    if results["radiomics"]["requires_radiologist_review"]:
        print("  ⚠️  Düşük güvenli bölgeler mevcut — Radyolog incelemesi gerekli!")
    
    # =====================================================================
    # ADIM 4: LONGİTUDİNAL ANALİZ — Registration + Hungarian Eşleştirme
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 4/6] Longitudinal Analiz: Registration + Lezyon Eşleştirme...")
    print(f"{'─' * 70}")
    
    longitudinal_result = run_longitudinal_analysis(
        baseline_ct=bl_final,
        baseline_mask=bl_mask,
        followup_ct=fu_final,
        followup_mask=fu_mask,
        output_dir=longitudinal_dir
    )
    
    results["longitudinal"] = longitudinal_result
    print(f"  [✓] Longitudinal analiz tamamlandı.")
    
    # =====================================================================
    # ADIM 5: RECIST 1.1 KARAR MOTORU — Deterministik Kural Motoru
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 5/6] RECIST 1.1 Karar Motoru: Deterministik Değerlendirme...")
    print(f"{'─' * 70}")
    
    recist_input = longitudinal_result.get("recist_input", {})
    sod_bl = recist_input.get("sod_baseline", 0.0)
    sod_fu = recist_input.get("sod_followup", 0.0)
    has_new = recist_input.get("new_lesion", False)
    
    # RECIST 1.1 Kuralları (Eisenhauer et al., 2009)
    if sod_bl > 0:
        change_pct = ((sod_fu - sod_bl) / sod_bl) * 100
    else:
        change_pct = 0.0
    
    if has_new:
        decision = "PD"
        reason = "Yeni lezyon tespit edildi. (iRECIST gereği teyit gerekebilir)"
    elif change_pct >= 20.0 and (sod_fu - sod_bl) >= 5.0:
        decision = "PD"
        reason = f"Tümör yükünde en az %20 ve mutlak en az 5mm artış (Değişim: %{change_pct:.1f})."
    elif sod_fu == 0:
        decision = "CR"
        reason = "Hedeflenen tüm lezyonlar tamamen kayboldu (Tam Yanıt)."
    elif change_pct <= -30.0:
        decision = "PR"
        reason = f"Tümör yükünde en az %30 azalma (Değişim: %{change_pct:.1f})."
    else:
        decision = "SD"
        reason = f"Belirgin değişiklik yok. Stabil durum (Değişim: %{change_pct:.1f})."
    
    results["recist_decision"] = {
        "baseline_sod": round(sod_bl, 2),
        "followup_sod": round(sod_fu, 2),
        "change_percentage": round(change_pct, 2),
        "new_lesion": has_new,
        "decision": decision,
        "explanation": reason
    }
    
    print(f"  KARAR: {decision}")
    print(f"  SOD: {sod_bl:.1f}mm → {sod_fu:.1f}mm ({change_pct:+.1f}%)")
    print(f"  Yeni Lezyon: {'EVET' if has_new else 'HAYIR'}")
    print(f"  [✓] RECIST 1.1 karar motoru tamamlandı.")
    
    # =====================================================================
    # ADIM 6: KLİNİK RAPOR ÜRETİMİ — Şablon veya LLM
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 6/6] Klinik Rapor Üretimi...")
    print(f"{'─' * 70}")
    
    # Radiomics sonuçlarını rapor için birleştir (follow-up ağırlıklı)
    report_radiomics = fu_radiomics
    
    report_input = {
        "segmentation": {
            "detected_lesions_count": fu_seg.get("detected_lesions_count", 0),
            "is_mock": fu_seg.get("is_mock", True)
        },
        "radiomics": report_radiomics,
        "longitudinal": longitudinal_result,
        "recist_decision": results["recist_decision"]
    }
    
    report_result = generate_clinical_report(
        pipeline_results=report_input,
        patient_info=patient_info,
        use_llm=use_llm,
        llm_api_key=llm_api_key
    )
    
    results["report"] = report_result
    
    # Raporu diske kaydet
    report_path = os.path.join(reports_dir, f"recist_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_result["report_text"])
    results["report"]["saved_to"] = report_path
    
    print(f"  Rapor kaynağı: {report_result['report_source']}")
    print(f"  Rapor dosyası: {report_path}")
    print(f"  [✓] Rapor üretimi tamamlandı.")
    
    # Süre hesapla
    elapsed = (datetime.now() - start_time).total_seconds()
    results["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["elapsed_seconds"] = round(elapsed, 2)
    
    print(f"\n{'=' * 70}")
    print(f"  UÇTAN UCA BOR HATTI TAMAMLANDI")
    print(f"  Karar: {decision} | Süre: {elapsed:.1f}s | Mock: {bl_seg['is_mock']}")
    print(f"{'=' * 70}\n")
    
    return results


def run_single_timepoint_pipeline(dicom_dir: str,
                                    output_dir: str = "./output",
                                    model_dir: str = "./models",
                                    patient_info: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Tek zamanlı (single timepoint) analiz boru hattı.
    Sadece bir CT çekimi olduğunda (baseline veya ilk çekim) kullanılır.
    
    DICOM → Preprocessing → Segmentasyon → Radiomics → Rapor
    (Longitudinal analiz ve RECIST kararı yapılamaz — karşılaştırma yok)
    """
    print("=" * 70)
    print("  TEK ZAMANLI ANALİZ BOR HATTI (Longitudinal karşılaştırma yok)")
    print("=" * 70)
    
    preprocess_dir = os.path.join(output_dir, "preprocessed")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Preprocessing
    print("\n[ADIM 1/3] Preprocessing...")
    raw = convert_dicom_to_nifti(dicom_dir, preprocess_dir, "raw_study.nii.gz")
    raw_hu = os.path.join(preprocess_dir, "raw_hu_study.nii.gz")
    resample_image(raw, raw_hu, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    windowed = os.path.join(preprocess_dir, "windowed_study.nii.gz")
    apply_windowing(raw, windowed)
    final = os.path.join(preprocess_dir, "study.nii.gz")
    resample_image(windowed, final, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    for tmp in [raw, windowed]:
        if os.path.exists(tmp):
            os.remove(tmp)
    
    # Segmentasyon
    print("\n[ADIM 2/3] AI Segmentasyon...")
    mask_path = os.path.join(masks_dir, "study_mask.nii.gz")
    seg_result = run_segmentation_inference(final, mask_path, model_dir)
    
    # Radiomics
    print("\n[ADIM 3/3] Radiomics Sınıflandırma...")
    radiomics_result = analyze_ct_and_mask_radiomics(raw_hu, mask_path)
    
    print(f"\n[✓] Tek zamanlı analiz tamamlandı.")
    print(f"    Lezyon sayısı: {seg_result.get('detected_lesions_count', 0)}")
    print(f"    Radyolog incelemesi gerekli: {radiomics_result.get('requires_radiologist_review', False)}")
    print(f"    NOT: RECIST 1.1 kararı üretilemedi (karşılaştırma için ikinci çekim gerekli).\n")
    
    return {
        "status": "success",
        "mode": "single_timepoint",
        "preprocessing": {"nifti_path": final, "raw_hu_path": raw_hu},
        "segmentation": seg_result,
        "radiomics": radiomics_result,
        "note": "RECIST 1.1 kararı için baseline ve follow-up olmak üzere en az iki çekim gereklidir."
    }


if __name__ == "__main__":
    print("=== Uçtan Uca Orkestrasyon Modülü (pipeline.py) Yüklendi ===")
    print("\nKullanım:")
    print("  from pipeline import run_full_pipeline, run_single_timepoint_pipeline")
    print("  result = run_full_pipeline('./baseline_dicom', './followup_dicom')")
