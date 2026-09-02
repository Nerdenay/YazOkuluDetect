"""
pipeline.py
===========
HepaRECIST-AI: Uçtan Uca 4D Longitudinal Analiz & RECIST 1.1 Karar Destek Boru Hattı

Tek bir çağrıyla tüm analiz sürecini yönetir:
  DICOM (t0 + t1) 
    → Preprocessing (1.0 mm³ Ham HU İzotropik Resampling)
    → AI Multi-Organ & Lezyon Segmentasyonu (nnU-Net v2 / Güvenli Mock, Label 8)
    → Radiomics Doku Doğrulama & Sphericity (Malign / Benign / Vasküler)
    → 4D Longitudinal B-Spline Registration & Hungarian Eşleştirme
    → RECIST 1.1 Deterministik Karar Motoru (Nadir ve Yeni Lezyon Korumalı)
    → Klinik Rapor Üretimi (Düzce Üniversitesi Formatı / LLM)
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional

# Proje Alt Modülleri
from preprocess import convert_dicom_to_nifti, resample_image
from inference import run_segmentation_inference
from radiomics_module import analyze_ct_and_mask_radiomics
from matching_engine import run_longitudinal_analysis
from report_generator import generate_clinical_report


def run_full_pipeline(baseline_dicom_dir: str, followup_dicom_dir: str,
                       output_dir: str = "./output",
                       model_dir: str = "./models",
                       patient_info: Optional[Dict[str, str]] = None,
                       lesion_label_id: int = 8,
                       use_llm: bool = False,
                       llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    İki zaman noktalı (t0 ve t1) uçtan uca analiz boru hattı.
    """
    print("=" * 70)
    print(" 🚀 HepaRECIST-AI: UÇTAN UCA LONGİTUDİNAL ANALİZ BORU HATTI")
    print("=" * 70)
    start_time = datetime.now()

    # Dizinleri hazırla
    preprocess_dir = os.path.join(output_dir, "preprocessed")
    masks_dir = os.path.join(output_dir, "masks")
    longitudinal_dir = os.path.join(output_dir, "longitudinal")
    reports_dir = os.path.join(output_dir, "reports")

    for d in [preprocess_dir, masks_dir, longitudinal_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    results = {
        "pipeline_version": "1.1.0",
        "started_at": start_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # =========================================================================
    # ADIM 1: PREPROCESSING (Ham HU ile 1.0mm İzotropik Resampling)
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 1/6] Ön İşleme: DICOM → NIfTI Dönüşümü & İzotropik Resampling...")
    print(f"{'─' * 70}")

    # t0 Baseline Preprocessing
    print("  [t0] Baseline işleniyor...")
    bl_tmp_raw = os.path.join(preprocess_dir, "tmp_raw_baseline.nii.gz")
    bl_final = os.path.join(preprocess_dir, "baseline.nii.gz")
    
    convert_dicom_to_nifti(baseline_dicom_dir, preprocess_dir, "tmp_raw_baseline.nii.gz")
    resample_image(bl_tmp_raw, bl_final, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    if os.path.exists(bl_tmp_raw):
        os.remove(bl_tmp_raw)

    # t1 Follow-up Preprocessing
    print("  [t1] Follow-up işleniyor...")
    fu_tmp_raw = os.path.join(preprocess_dir, "tmp_raw_followup.nii.gz")
    fu_final = os.path.join(preprocess_dir, "followup.nii.gz")
    
    convert_dicom_to_nifti(followup_dicom_dir, preprocess_dir, "tmp_raw_followup.nii.gz")
    resample_image(fu_tmp_raw, fu_final, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    if os.path.exists(fu_tmp_raw):
        os.remove(fu_tmp_raw)

    results["preprocessing"] = {
        "baseline_nifti": bl_final,
        "followup_nifti": fu_final
    }
    print("  [✓] Ön işleme tamamlandı (Ham HU değerleri korundu).")

    # =========================================================================
    # ADIM 2: AI SEGMENTASYON (Multi-Organ + Label 8 Lezyon)
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 2/6] AI Segmentasyon: nnU-Net v2 ile 3D Doku & Lezyon Çıkarımı...")
    print(f"{'─' * 70}")

    bl_mask = os.path.join(masks_dir, "baseline_mask.nii.gz")
    fu_mask = os.path.join(masks_dir, "followup_mask.nii.gz")

    print("  [t0] Baseline segmentasyonu...")
    bl_seg = run_segmentation_inference(bl_final, bl_mask, model_dir, lesion_label_id=lesion_label_id)
    print("  [t1] Follow-up segmentasyonu...")
    fu_seg = run_segmentation_inference(fu_final, fu_mask, model_dir, lesion_label_id=lesion_label_id)

    results["segmentation"] = {
        "baseline": bl_seg,
        "followup": fu_seg,
        "is_mock": bl_seg["is_mock"]
    }
    print(f"  [✓] Segmentasyon tamamlandı (Mock Modu: {bl_seg['is_mock']}).")

    # =========================================================================
    # ADIM 3: RADYOMİK SINIFLANDIRMA (Doğrulama, Doku Heterojenliği & Sphericity)
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 3/6] Radyomik Doğrulama: HU, Heterojenlik ve 3D Küresellik (Ψ)...")
    print(f"{'─' * 70}")

    print("  [t0] Baseline radyomik analizi...")
    bl_radiomics = analyze_ct_and_mask_radiomics(bl_final, bl_mask, lesion_label_id=lesion_label_id)
    print("  [t1] Follow-up radyomik analizi...")
    fu_radiomics = analyze_ct_and_mask_radiomics(fu_final, fu_mask, lesion_label_id=lesion_label_id)

    results["radiomics"] = {
        "baseline": bl_radiomics,
        "followup": fu_radiomics,
        "requires_radiologist_review": (
            bl_radiomics.get("requires_radiologist_review", False) or
            fu_radiomics.get("requires_radiologist_review", False)
        )
    }
    print("  [✓] Radyomik doku analizi tamamlandı.")
    if results["radiomics"]["requires_radiologist_review"]:
        print("  ⚠️  Düşük güvenli lezyon tespit edildi — Hekim onay bayrağı aktif!")

    # =========================================================================
    # ADIM 4: LONGİTUDİNAL ANALİZ (B-Spline Registration & Hungarian Takip)
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 4/6] Longitudinal Takip: B-Spline Registration & Hungarian Eşleştirme...")
    print(f"{'─' * 70}")

    longitudinal_result = run_longitudinal_analysis(
        baseline_ct=bl_final,
        baseline_mask=bl_mask,
        followup_ct=fu_final,
        followup_mask=fu_mask,
        output_dir=longitudinal_dir
    )

    results["longitudinal"] = longitudinal_result
    print("  [✓] 4D Longitudinal takip ve lezyon eşleştirme tamamlandı.")

    # =========================================================================
    # ADIM 5: RECIST 1.1 DETERMINİSTİK KARAR MOTORU
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 5/6] RECIST 1.1 Karar Motoru: Matematiksel Değerlendirme...")
    print(f"{'─' * 70}")

    # Sözlük anahtarı uyumluluğu (recist_metrics veya recist_input)
    recist_data = longitudinal_result.get("recist_metrics", longitudinal_result.get("recist_input", {}))
    sod_bl = recist_data.get("sod_baseline", 0.0)
    sod_fu = recist_data.get("sod_followup", 0.0)
    has_new = recist_data.get("has_new_lesions", recist_data.get("new_lesion", False))

    # RECIST 1.1 Kuralları (Eisenhauer et al., 2009)
    change_pct = ((sod_fu - sod_bl) / sod_bl * 100) if sod_bl > 0 else 0.0

    if has_new:
        decision = "PD"
        reason = "Yeni lezyon tespit edildi. Kesin Progresif Hastalık (PD)."
    elif change_pct >= 20.0 and (sod_fu - sod_bl) >= 5.0:
        decision = "PD"
        reason = f"Tümör yükünde en az %20 ve mutlak en az 5mm artış tespit edildi (Değişim: %{change_pct:.1f})."
    elif sod_bl > 0 and sod_fu == 0.0:
        decision = "CR"
        reason = "Hedeflenen tüm lezyonlar tamamen kayboldu (Tam Yanıt - CR)."
    elif change_pct <= -30.0:
        decision = "PR"
        reason = f"Tümör yükünde en az %30 azalma tespit edildi (Değişim: %{change_pct:.1f})."
    else:
        decision = "SD"
        reason = f"Belirgin küçülme veya progresyon kriteri karşılanmadı. Stabil durum (SD). (Değişim: %{change_pct:.1f})"

    results["recist_decision"] = {
        "baseline_sod": round(sod_bl, 2),
        "followup_sod": round(sod_fu, 2),
        "change_percentage": round(change_pct, 2),
        "new_lesion": has_new,
        "decision": decision,
        "explanation": reason
    }

    print(f"  KLİNİK KARAR : {decision}")
    print(f"  SOD DEĞİŞİMİ : {sod_bl:.1f} mm → {sod_fu:.1f} mm ({change_pct:+.1f}%)")
    print(f"  YENİ LEZYON  : {'EVET (PD)' if has_new else 'HAYIR'}")
    print("  [✓] RECIST 1.1 karar motoru sonuçlandı.")

    # =========================================================================
    # ADIM 6: KLİNİK RAPOR ÜRETİMİ
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("[ADIM 6/6] Klinik Rapor Üretimi (Düzce Üniversitesi Formatı)...")
    print(f"{'─' * 70}")

    report_input = {
        "segmentation": {
            "detected_lesions_count": fu_seg.get("detected_lesions_count", 0),
            "is_mock": fu_seg.get("is_mock", True)
        },
        "radiomics": fu_radiomics,
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

    # Raporu diske yaz
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(reports_dir, f"recist_report_{timestamp_str}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_result.get("report_text", ""))
    results["report"]["saved_to"] = report_file

    elapsed = (datetime.now() - start_time).total_seconds()
    results["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["elapsed_seconds"] = round(elapsed, 2)

    print(f"  Rapor Kaynağı : {report_result.get('report_source', 'Standart')}")
    print(f"  Rapor Dosyası : {report_file}")
    print(f"\n{'=' * 70}")
    print(f" 🎉 BORU HATTI BAŞARIYLA TAMAMLANDI")
    print(f"  Nihai Karar : {decision} | Toplam Süre: {elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    return results


def run_single_timepoint_pipeline(dicom_dir: str,
                                    output_dir: str = "./output",
                                    model_dir: str = "./models",
                                    lesion_label_id: int = 8,
                                    patient_info: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Tek zamanlı (tek BT çekimi) analiz boru hattı.
    """
    print("=" * 70)
    print(" 🔎 TEK ZAMANLI BT ANALİZ BORU HATTI")
    print("=" * 70)

    preprocess_dir = os.path.join(output_dir, "preprocessed")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 1. Ön İşleme (Ham HU korunur)
    print("\n[ADIM 1/3] Ön İşleme...")
    tmp_raw = os.path.join(preprocess_dir, "tmp_single_raw.nii.gz")
    final_ct = os.path.join(preprocess_dir, "study.nii.gz")
    
    convert_dicom_to_nifti(dicom_dir, preprocess_dir, "tmp_single_raw.nii.gz")
    resample_image(tmp_raw, final_ct, new_spacing=(1.0, 1.0, 1.0), is_label=False)
    if os.path.exists(tmp_raw):
        os.remove(tmp_raw)

    # 2. Segmentasyon
    print("\n[ADIM 2/3] AI Segmentasyon...")
    mask_path = os.path.join(masks_dir, "study_mask.nii.gz")
    seg_result = run_segmentation_inference(final_ct, mask_path, model_dir, lesion_label_id=lesion_label_id)

    # 3. Radyomik
    print("\n[ADIM 3/3] Radyomik Doku Analizi...")
    radiomics_result = analyze_ct_and_mask_radiomics(final_ct, mask_path, lesion_label_id=lesion_label_id)

    print(f"\n[✓] Tekil analiz tamamlandı. Tespit edilen lezyon: {seg_result.get('detected_lesions_count', 0)}")
    return {
        "status": "success",
        "mode": "single_timepoint",
        "ct_path": final_ct,
        "mask_path": mask_path,
        "segmentation": seg_result,
        "radiomics": radiomics_result,
        "note": "RECIST 1.1 tedavi yanıtı için en az iki zamanlı (t0, t1) çekim gereklidir."
    }


if __name__ == "__main__":
    print("=== HepaRECIST-AI: Uçtan Uca Boru Hattı Orkestratörü (pipeline.py) ===")
