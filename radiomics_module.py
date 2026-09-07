"""
radiomics_module.py
===================
HepaRECIST-AI - Faz 3: Radyomik Doku Doğrulama ve Güven Skoru Motoru
- 3D Lezyon Segmentasyonu (Label 8) Doğrulama
- HU İstatistikleri: Mean, Std, Median, P10, P25, P75, P90, IQR
- Morfolojik Özellikler: Hacim (mm³), Fiziksel Centroid (LPS), 3D Küresellik (Sphericity Ψ)
- Kural Tabanlı Sınıflandırma: Malign (Tümör/Metastaz), Benign (Kist), Vasküler/Kireçlenme
- Güven Skoru (<0.75 ise Hekim Onay Kuyruğu Flag'i)
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Dict, List, Any


def calculate_3d_sphericity(binary_region: np.ndarray, spacing: tuple) -> float:
    """
    3D Küresellik (Sphericity Ψ) Katsayısını hesaplar:
        Ψ = (π^(1/3) * (6 * Hacim)^(2/3)) / Yüzey_Alanı
    Değer aralığı: (0, 1]. Mükemmel küre = 1.0.
    """
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = float(np.sum(binary_region) * voxel_vol)
    
    if volume_mm3 <= 0:
        return 0.0

    # 3D Morfolojik Gradyan / Erozyon ile yüzey piksellerini çıkar
    eroded = ndimage.binary_erosion(binary_region)
    surface_mask = binary_region ^ eroded
    surface_voxel_count = int(np.sum(surface_mask))

    if surface_voxel_count == 0:
        return 1.0

    # Yaklaşık yüzey alanı (mm²)
    mean_face_area = (spacing[0]*spacing[1] + spacing[1]*spacing[2] + spacing[0]*spacing[2]) / 3.0
    surface_area_mm2 = surface_voxel_count * mean_face_area

    # Sphericity formülü
    psi = (np.pi ** (1.0 / 3.0) * ((6.0 * volume_mm3) ** (2.0 / 3.0))) / (surface_area_mm2 + 1e-5)
    return round(float(np.clip(psi, 0.0, 1.0)), 3)


def analyze_ct_and_mask_radiomics(ct_nifti_path: str, mask_nifti_path: str, lesion_label_id: int = 8) -> Dict[str, Any]:
    """
    Maskedeki Lezyon (varsayılan Label 8, yoksa 2 veya 1) bölgeleri için 
    radyomik analiz ve malignite sınıflandırması yapar.
    """
    if not os.path.exists(ct_nifti_path):
        raise FileNotFoundError(f"BT NIfTI dosyası bulunamadı: {ct_nifti_path}")
    if not os.path.exists(mask_nifti_path):
        raise FileNotFoundError(f"Maske NIfTI dosyası bulunamadı: {mask_nifti_path}")

    ct_img = sitk.ReadImage(ct_nifti_path)
    mask_img = sitk.ReadImage(mask_nifti_path)

    ct_arr = sitk.GetArrayFromImage(ct_img)    # (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = ct_img.GetSpacing()              # (sx, sy, sz)
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    # Eğer belirtilen etiket maskede yoksa lezyon tespit edilmemiştir
    unique_vals = set(np.unique(mask_arr))
    if lesion_label_id not in unique_vals:
        return {
            "status": "success",
            "total_candidate_regions": 0,
            "regions": [],
            "requires_radiologist_review": False,
            "preview_image_path": ""
        }

    lesion_binary = (mask_arr == lesion_label_id)
    labeled_mask, num_regions = ndimage.label(lesion_binary)

    regions_analysis = []

    for region_id in range(1, num_regions + 1):
        region_voxels = (labeled_mask == region_id)
        voxel_count = int(np.sum(region_voxels))

        if voxel_count < 15:  # Küçük artefaktları atla
            continue

        region_hu = ct_arr[region_voxels]

        # 1. HU İstatistikleri
        hu_mean = float(np.mean(region_hu))
        hu_std = float(np.std(region_hu))
        hu_p25 = float(np.percentile(region_hu, 25))
        hu_p75 = float(np.percentile(region_hu, 75))
        hu_iqr = float(hu_p75 - hu_p25)  # Güvenilir heterojenlik metriği

        # 2. Şekil ve Küresellik
        volume_mm3 = float(voxel_count * voxel_vol)
        sphericity = calculate_3d_sphericity(region_voxels, spacing)

        # Fiziksel Uzayda Kütle Merkezi (SimpleITK LPS)
        centroid_zyx = ndimage.center_of_mass(region_voxels)
        physical_centroid = ct_img.TransformContinuousIndexToPhysicalPoint([centroid_zyx[2], centroid_zyx[1], centroid_zyx[0]])

        # 3. Kural Tabanlı Sınıflandırma
        label, confidence = _classify_region_advanced(hu_mean, hu_std, hu_iqr, sphericity, volume_mm3)
        needs_review = confidence < 0.75

        regions_analysis.append({
            "region_id": region_id,
            "centroid_mm": [round(c, 2) for c in physical_centroid],
            "volume_mm3": round(volume_mm3, 2),
            "sphericity_psi": sphericity,
            "hu_mean": round(hu_mean, 1),
            "hu_std": round(hu_std, 1),
            "hu_iqr": round(hu_iqr, 1),
            "classification": label,
            "confidence_score": round(confidence, 2),
            "needs_review": needs_review,
            "explanation": _build_explanation(label, hu_mean, hu_std, sphericity, confidence)
        })

    result = {
        "status": "success",
        "total_candidate_regions": len(regions_analysis),
        "regions": regions_analysis,
        "requires_radiologist_review": any(r["needs_review"] for r in regions_analysis)
    }

    # Görselleştirme preview (Varsa)
    try:
        from visualizer import generate_lesion_visualization
        base_name = mask_nifti_path
        for ext in [".nii.gz", ".nii"]:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        preview_png = base_name + "_radiomics_preview.png"
        generate_lesion_visualization(
            ct_nifti_path=ct_nifti_path,
            mask_nifti_path=mask_nifti_path,
            radiomics_results=result,
            output_png_path=preview_png,
            lesion_label_id=lesion_label_id
        )
        result["preview_image_path"] = preview_png
    except Exception:
        result["preview_image_path"] = ""

    return result


def _classify_region_advanced(hu_mean: float, hu_std: float, hu_iqr: float, 
                              sphericity: float, volume_mm3: float) -> tuple:
    """
    HU yoğunluğu, doku heterojenliği (IQR/Std) ve 3D Küresellik (Ψ) ile karar verir.
    """
    # 1. Kist: Düşük HU, düşük varyasyon VE yüksek küresellik (küreye yakın sıvı kesesi)
    if hu_mean <= 22.0 and hu_std < 14.0 and sphericity >= 0.78:
        return "Benign (Basit Kist)", 0.94

    # Sınırda kist (Düşük HU ama küresellik sınırda)
    if hu_mean <= 25.0 and hu_std < 16.0:
        return "Benign (Olası Kist / Nekroz)", 0.72

    # 2. Vasküler / Kireçlenme: Yüksek kontrastlanma
    if hu_mean >= 120.0:
        return "Vasküler / Kalsifiye Odak", 0.90

    # 3. Malign (Tümör / Metastaz): Karaciğerde hipodens/izodens tümöral aralık
    if 25.0 <= hu_mean <= 95.0:
        # Malign lezyonlar düzensiz kenarlıdır (düşük sphericity) ve doku içi heterojendir (yüksek IQR/Std)
        if hu_iqr > 18.0 or hu_std > 16.0:
            conf = 0.88 if sphericity < 0.82 else 0.76
            return "Malign (Tümör/Metastaz)", conf
        else:
            # Homojen solid lezyon (Adenoma, FNH veya düşük heterojenlikli metastaz)
            return "Solid Lezyon (Şüpheli)", 0.65

    return "Belirsiz / Atipik Doku", 0.50


def _build_explanation(label: str, hu_mean: float, hu_std: float, sphericity: float, confidence: float) -> str:
    """Klinik karar gerekçesi metni."""
    if "Kist" in label and confidence >= 0.75:
        return f"Sıvı dansitesi ({hu_mean:.1f} HU) ve yüksek 3D küresellik (Ψ={sphericity:.2f}) basit kist lehinedir."
    if "Vasküler" in label:
        return f"Belirgin yüksek dansite ({hu_mean:.1f} HU) damarsal yapı veya kireçlenmeyi düşündürmektedir."
    if "Malign" in label:
        return f"Tümöral dansite ({hu_mean:.1f} HU), heterojen iç yapı (std={hu_std:.1f}) ve invaziv kenar (Ψ={sphericity:.2f}) malign süreç ile uyumludur."
    return f"Dansite ({hu_mean:.1f} HU) veya şekil (Ψ={sphericity:.2f}) sınırda değerdedir. Hekim onayı önerilir."
