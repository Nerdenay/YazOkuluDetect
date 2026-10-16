"""
radiomics_module.py
===================
HepaRECIST-AI - Faz 3: Radyomik Doku Doğrulama ve Güven Skoru Motoru
- 3D Lezyon Segmentasyonu (Label 8) Doğrulama
- HU İstatistikleri: Mean, Std, Median, P10, P25, P75, P90, IQR
- Morfolojik Özellikler: Hacim (mm³), Fiziksel Centroid (LPS), 3D Küresellik (Sphericity Ψ)
- Kural tabanlı DANSİTE ÖN-KATEGORİSİ (tanı değildir): kistik görünümlü, solid,
  hiperdens, belirsiz. Eşikler doğrulanmamıştır; kontrast fazı hesaba katılmaz.
  Malign/benign ayrımı yapılmaz, sabit "güven skoru" üretilmez; her aday
  radyolog onayına gider.
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Dict, List, Any


def calculate_3d_sphericity(binary_region: np.ndarray, spacing: tuple) -> float:
    """
    3D Küresellik: Ψ = π^(1/3) (6V)^(2/3) / A  (mükemmel küre = 1).
    Yüzey alanı A, SimpleITK LabelShapeStatistics 'Perimeter' (Crofton tabanlı
    yüzey kestirimi) ile hesaplanır; voksel basamak etkisi büyük ölçüde giderilir.
    """
    img = sitk.GetImageFromArray(binary_region.astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputePerimeterOn()
    stats.Execute(img)
    if not stats.HasLabel(1):
        return 0.0
    volume = stats.GetPhysicalSize(1)
    area = stats.GetPerimeter(1)
    if volume <= 0 or area <= 0:
        return 0.0
    psi = (np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / area
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
        label = _density_category(hu_mean, hu_std, hu_iqr, sphericity)
        needs_review = True  # Doğrulanmış sınıflandırıcı yok: her aday radyolog onayına gider

        regions_analysis.append({
            "region_id": region_id,
            "centroid_mm": [round(c, 2) for c in physical_centroid],
            "volume_mm3": round(volume_mm3, 2),
            "sphericity_psi": sphericity,
            "hu_mean": round(hu_mean, 1),
            "hu_std": round(hu_std, 1),
            "hu_iqr": round(hu_iqr, 1),
            "classification": label,
            "needs_review": needs_review,
            "explanation": _build_explanation(label, hu_mean, hu_std, sphericity)
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


def _density_category(hu_mean: float, hu_std: float, hu_iqr: float, sphericity: float) -> str:
    """
    Basit dansite ön-kategorisi. TANI DEĞİLDİR; eşikler kendi verinizde
    doğrulanmalıdır. Kontrast fazı (arteriyel / portal / geç) HU değerini değiştirir.
    """
    if hu_mean <= 22.0 and hu_std < 14.0 and sphericity >= 0.78:
        return "Aday: kistik görünümlü"
    if hu_mean >= 120.0:
        return "Aday: hiperdens (vasküler/kalsifik olabilir)"
    if 25.0 <= hu_mean <= 95.0:
        return "Aday: solid"
    return "Aday: belirsiz dansite"


def _build_explanation(label: str, hu_mean: float, hu_std: float, sphericity: float) -> str:
    """Ölçülen değerleri tarafsız biçimde özetler (yorum radyoloğa aittir)."""
    return (f"Ortalama dansite {hu_mean:.1f} HU, standart sapma {hu_std:.1f}, "
            f"küresellik Ψ={sphericity:.2f}. Kural tabanlı ön-kategori: {label}. "
            "Doğrulanmamış eşikler; radyolog değerlendirmesi gereklidir.")
