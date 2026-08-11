import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from typing import Dict, List, Any


def analyze_ct_and_mask_radiomics(ct_nifti_path: str, mask_nifti_path: str) -> Dict[str, Any]:
    """
    AI segmentasyon maskesindeki her bir 3D lezyon adayı için Radyomik özellik çıkarımı
    ve kural tabanlı sınıflandırma yapar.

    Lezyon adayları üç sınıfa ayrılır:
        Malign (Tümör/Metastaz) : Orta-yüksek HU (25-95 HU), yüksek doku heterojenliği.
        Benign (Kist)           : Düşük HU (0-22 HU), homojen sıvı yapısı.
        Vasküler / Kireçlenme   : Çok yüksek HU (≥120 HU).

    Çıkarılan radyomik özellikler:
        - Yoğunluk (HU) istatistikleri: mean, std, min, max, P25, P75.
        - Şekil: Kütle merkezi (centroid), hacim (mm³).
        - Doku heterojenliği: Normalize standart sapma (hu_std / |hu_mean|).

    Güven skoru <0.75 ise lezyon adayı otomatik olarak Radyolog Onay Kuyruğu'na alınır.

    Args:
        ct_nifti_path   : Ham HU değerlerini içeren BT NIfTI dosyasının yolu.
                          (preprocess.py çıktısı: raw_hu_*.nii.gz)
        mask_nifti_path : AI segmentasyon maskesi NIfTI dosyasının yolu.

    Returns:
        status                      : "success"
        total_candidate_regions     : Analiz edilen lezyon adayı sayısı.
        regions                     : Her lezyon için sınıflandırma sonuçları listesi.
        requires_radiologist_review : En az bir düşük güvenli lezyon varsa True.
        preview_image_path          : Üretilen 2D BT kesit PNG görselinin yolu.
    """
    if not os.path.exists(ct_nifti_path):
        raise FileNotFoundError(f"BT NIfTI dosyası bulunamadı: {ct_nifti_path}")
    if not os.path.exists(mask_nifti_path):
        raise FileNotFoundError(f"Maske NIfTI dosyası bulunamadı: {mask_nifti_path}")

    ct_img = sitk.ReadImage(ct_nifti_path)
    mask_img = sitk.ReadImage(mask_nifti_path)

    ct_arr = sitk.GetArrayFromImage(ct_img)     # Eksen: (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask_img)  # Eksen: (Z, Y, X)
    spacing = ct_img.GetSpacing()               # (x_mm, y_mm, z_mm)

    # Maskedeki birbirinden bağımsız 3D lezyon adaylarını etiketle
    labeled_mask, num_regions = ndimage.label(mask_arr > 0)

    regions_analysis = []

    for region_id in range(1, num_regions + 1):
        region_voxels = (labeled_mask == region_id)
        voxel_count = int(np.sum(region_voxels))

        # 10 voxel'den küçük alanlar muhtemelen gürültüdür, atla
        if voxel_count < 10:
            continue

        region_hu = ct_arr[region_voxels]

        # --- 1. HU Yoğunluk İstatistikleri ---
        hu_mean = float(np.mean(region_hu))
        hu_std = float(np.std(region_hu))

        # --- 2. Şekil Özellikleri ---
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = float(voxel_count * voxel_vol_mm3)

        centroid_voxels = ndimage.center_of_mass(region_voxels)
        centroid_mm = [
            float(centroid_voxels[2] * spacing[0]),  # X ekseni
            float(centroid_voxels[1] * spacing[1]),  # Y ekseni
            float(centroid_voxels[0] * spacing[2])   # Z ekseni
        ]

        # --- 3. Doku Heterojenliği ---
        # Normalize standart sapma: yüksek değer = heterojen doku = malign lehine
        texture_entropy = float(hu_std / (abs(hu_mean) + 1e-5))

        # --- 4. Kural Tabanlı Sınıflandırma ---
        label, confidence = _classify_region(hu_mean, hu_std, texture_entropy, volume_mm3)

        # Güven skoru %75 altındaysa radyolog incelemesi gerekir
        needs_review = confidence < 0.75

        regions_analysis.append({
            "region_id": region_id,
            "centroid_mm": [round(c, 2) for c in centroid_mm],
            "volume_mm3": round(volume_mm3, 2),
            "hu_mean": round(hu_mean, 1),
            "hu_std": round(hu_std, 1),
            "classification": label,
            "confidence_score": round(confidence, 2),
            "needs_review": needs_review,
            "explanation": _build_explanation(label, hu_mean, hu_std, confidence)
        })

    result = {
        "status": "success",
        "total_candidate_regions": len(regions_analysis),
        "regions": regions_analysis,
        "requires_radiologist_review": any(r["needs_review"] for r in regions_analysis)
    }

    # Radiomics sonuçlarıyla birlikte 2D BT kesit görselini üret
    try:
        from visualizer import generate_lesion_visualization
        preview_png = mask_nifti_path.replace(".nii.gz", "_radiomics_preview.png")
        generate_lesion_visualization(
            ct_nifti_path=ct_nifti_path,
            mask_nifti_path=mask_nifti_path,
            radiomics_results=result,
            output_png_path=preview_png
        )
        result["preview_image_path"] = preview_png
    except Exception as exc:
        print(f"[WARNING] Radiomics kesit görseli üretilemedi: {exc}")
        result["preview_image_path"] = ""

    return result


def _classify_region(
    hu_mean: float,
    hu_std: float,
    texture_entropy: float,
    volume_mm3: float
) -> tuple:
    """
    Radyomik özelliklere göre üç sınıftan birini ve güven skorunu döndürür.

    Sınıflandırma mantığı:
        Kist     : hu_mean ≤ 22 HU VE hu_std < 12 (homojen sıvı)
        Vasküler : hu_mean ≥ 120 HU (kontrast madde veya kireçlenme)
        Malign   : 25 ≤ hu_mean ≤ 95 HU (tümöral yoğunluk aralığı)
                   -> Yüksek std (>16) varsa güçlü malign, yoksa sınırda malign.
        Belirsiz : Yukarıdakilerin dışında kalan alanlar, radyolog onayı gerekir.

    Returns:
        (sınıf_etiketi, güven_skoru) tuple'ı.
    """
    if hu_mean <= 22.0 and hu_std < 12.0:
        return "Benign (Kist)", 0.92

    if hu_mean >= 120.0:
        return "Vasküler / Kireçlenme", 0.88

    if 25.0 <= hu_mean <= 95.0:
        if hu_std > 16.0:
            return "Malign (Tümör/Metastaz)", 0.89
        else:
            return "Malign (Şüpheli)", 0.68

    return "Belirsiz / Karışık Yapı", 0.55


def _build_explanation(label: str, hu_mean: float, hu_std: float, confidence: float) -> str:
    """Sınıflandırma kararına açıklama metni üretir."""
    if "Benign" in label:
        return f"Düşük yoğunluk ({hu_mean:.1f} HU) ve homojen doku yapısı kist ile uyumludur."
    if "Vasküler" in label:
        return f"Yüksek kontrastlanma ({hu_mean:.1f} HU) damarsal veya kalsifiye yapıyı düşündürmektedir."
    if "Malign" in label and confidence >= 0.75:
        return f"Düzensiz doku dağılımı (std={hu_std:.1f} HU) ve tümöral yoğunluk ({hu_mean:.1f} HU) malign süreç lehinedir."
    return f"Sınırda değerler ({hu_mean:.1f} HU). Kesin değerlendirme için radyolog onayı gerekmektedir."


if __name__ == "__main__":
    print("=== Radiomics Sınıflandırma Modülü (radiomics_module.py) ===")
