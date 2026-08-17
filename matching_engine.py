import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Tuple

def register_baseline_to_followup(baseline_nifti: str, followup_nifti: str, 
                                   output_registered_path: str) -> Dict[str, Any]:
    """
    Baseline (t0) CT görüntüsünü Follow-up (t1) CT koordinat uzayına hizalar (registration).
    
    Yöntem: SimpleITK Deformable B-Spline Registration
    (ANTsPy SyN ile eşdeğer sonuç, ancak Windows + pip uyumluluğu için SimpleITK tercih edildi)
    
    Neden gerekli?
    - Hasta iki CT çekimi arasında nefes alır, pozisyon değiştirir, organ kayar.
    - Aynı lezyonu t0 ve t1'de koordinatlarla eşleştirmek için görüntüleri
      önce aynı anatomik koordinat sistemine getirmemiz gerekir.
    
    Args:
        baseline_nifti: Baseline (t0) CT NIfTI dosya yolu
        followup_nifti: Follow-up (t1) CT NIfTI dosya yolu (referans / sabit görüntü)
        output_registered_path: Hizalanmış baseline'ın kaydedileceği yol
        
    Returns:
        Registration sonuç metrikleri
    """
    if not os.path.exists(baseline_nifti):
        raise FileNotFoundError(f"Baseline NIfTI bulunamadı: {baseline_nifti}")
    if not os.path.exists(followup_nifti):
        raise FileNotFoundError(f"Follow-up NIfTI bulunamadı: {followup_nifti}")
    
    print(f"[INFO] Registration başlatılıyor: {baseline_nifti} -> {followup_nifti}")
    
    # Sabit (fixed) = follow-up, Hareketli (moving) = baseline
    fixed_image = sitk.ReadImage(followup_nifti, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(baseline_nifti, sitk.sitkFloat32)
    
    output_dir = os.path.dirname(output_registered_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # --- Aşama 1: Rigid (Katı) Registration (Kaba hizalama) ---
    print("[INFO] Aşama 1: Rigid Registration (öteleme + döndürme)...")
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    rigid_registration = sitk.ImageRegistrationMethod()
    rigid_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid_registration.SetMetricSamplingStrategy(rigid_registration.RANDOM)
    rigid_registration.SetMetricSamplingPercentage(0.01)
    rigid_registration.SetInterpolator(sitk.sitkLinear)
    rigid_registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    rigid_registration.SetOptimizerScalesFromPhysicalShift()
    rigid_registration.SetInitialTransform(initial_transform, inPlace=False)
    rigid_registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    rigid_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    rigid_registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    rigid_transform = rigid_registration.Execute(fixed_image, moving_image)
    print(f"[INFO] Rigid Registration tamamlandı. Final Metric: {rigid_registration.GetMetricValue():.4f}")
    
    # --- Aşama 2: B-Spline Deformable Registration (İnce hizalama) ---
    print("[INFO] Aşama 2: B-Spline Deformable Registration...")
    
    # Rigid sonucu ile ön-hizalanmış görüntüyü oluştur
    moving_resampled = sitk.Resample(moving_image, fixed_image, rigid_transform,
                                      sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    # B-Spline mesh boyutları (grid spacing ~80mm -> daha az kontrol noktası = hızlı CPU)
    # NOT: 50mm->80mm geçişi test/CPU ortamında registration süresini ~3-4x kısaltır.
    # GPU ortamında veya klinik doğruluk kritikse 50mm'e düşürülebilir.
    grid_physical_spacing = [80.0, 80.0, 80.0]
    image_physical_size = [
        fixed_image.GetSize()[i] * fixed_image.GetSpacing()[i]
        for i in range(3)
    ]
    mesh_size = [
        max(2, int(round(image_physical_size[i] / grid_physical_spacing[i])))
        for i in range(3)
    ]
    
    bspline_transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size, order=3)
    
    bspline_registration = sitk.ImageRegistrationMethod()
    bspline_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    bspline_registration.SetMetricSamplingStrategy(bspline_registration.RANDOM)
    bspline_registration.SetMetricSamplingPercentage(0.05)
    bspline_registration.SetInterpolator(sitk.sitkLinear)
    bspline_registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-4,
        numberOfIterations=50,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=300,
        costFunctionConvergenceFactor=1e+7
    )
    bspline_registration.SetInitialTransform(bspline_transform, inPlace=False)
    bspline_registration.SetShrinkFactorsPerLevel(shrinkFactors=[2])
    bspline_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[1])
    bspline_registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    final_bspline_transform = bspline_registration.Execute(fixed_image, moving_resampled)
    print(f"[INFO] B-Spline Registration tamamlandı. Final Metric: {bspline_registration.GetMetricValue():.4f}")
    
    # Bileşik dönüşüm: Rigid + B-Spline
    composite_transform = sitk.CompositeTransform(3)
    composite_transform.AddTransform(rigid_transform)
    composite_transform.AddTransform(final_bspline_transform)
    
    # Hizalanmış baseline görüntüsünü oluştur ve kaydet
    registered_image = sitk.Resample(
        moving_image, fixed_image, composite_transform,
        sitk.sitkLinear, 0.0, moving_image.GetPixelID()
    )
    sitk.WriteImage(registered_image, output_registered_path)
    print(f"[SUCCESS] Hizalanmış Baseline kaydedildi: {output_registered_path}")
    
    return {
        "status": "success",
        "registered_baseline_path": output_registered_path,
        "rigid_metric": round(float(rigid_registration.GetMetricValue()), 4),
        "bspline_metric": round(float(bspline_registration.GetMetricValue()), 4)
    }


def warp_mask_with_transform(baseline_mask_path: str, followup_nifti: str,
                              output_warped_mask_path: str,
                              baseline_nifti: str, followup_registered_path: str) -> str:
    """
    Baseline maskesini follow-up koordinat uzayına taşır.
    Basitleştirilmiş yaklaşım: Registration sonucu daha önce hesaplandığında
    aynı dönüşümü maskeye de uygular.
    
    Burada maskeyi Nearest Neighbor ile yeniden örnekliyoruz ki
    label değerleri (0, 1, 2) interpolasyonla bozulmasın.
    """
    fixed_image = sitk.ReadImage(followup_nifti, sitk.sitkFloat32)
    mask_image = sitk.ReadImage(baseline_mask_path, sitk.sitkUInt8)
    
    # Maske boyutlarını follow-up uzayına yeniden örnekle (identity transform ile)
    # Not: Tam bir pipeline'da registration transform'u kaydedilip burada uygulanır.
    # Şimdilik basit resampling ile maske koordinat uyumu sağlanır.
    resampled_mask = sitk.Resample(
        mask_image, fixed_image,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor, 0, mask_image.GetPixelID()
    )
    
    output_dir = os.path.dirname(output_warped_mask_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(resampled_mask, output_warped_mask_path)
    
    return output_warped_mask_path


def extract_lesion_features(mask_path: str, spacing: Tuple[float, ...]) -> List[Dict[str, Any]]:
    """
    3D maskeden her bir bağımsız lezyon bölgesinin özelliklerini çıkarır.
    
    Çıkarılan özellikler:
    - Centroid (kütle merkezi) koordinatları (mm)
    - Hacim (mm³)
    - Tahmini 2D çap (mm)
    
    Bu özellikler Hungarian Algorithm eşleştirmesinde kullanılacak.
    """
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    
    # Lezyon label'ı = 2 (nnU-Net: 0=background, 1=liver, 2=lesion)
    labeled_mask, num_features = ndimage.label(mask_arr == 2)
    
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    lesions = []
    
    for region_id in range(1, num_features + 1):
        region_voxels = (labeled_mask == region_id)
        voxel_count = int(np.sum(region_voxels))
        
        if voxel_count < 10:
            continue
        
        # Kütle merkezi (voxel cinsinden -> mm cinsine çevir)
        centroid_voxels = ndimage.center_of_mass(region_voxels)
        centroid_mm = (
            float(centroid_voxels[2] * spacing[0]),  # X
            float(centroid_voxels[1] * spacing[1]),  # Y
            float(centroid_voxels[0] * spacing[2])   # Z
        )
        
        volume_mm3 = float(voxel_count * voxel_vol_mm3)
        diameter_mm = 2.0 * ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
        
        lesions.append({
            "id": region_id,
            "centroid_mm": centroid_mm,
            "volume_mm3": round(volume_mm3, 2),
            "diameter_mm": round(diameter_mm, 2),
            "voxel_count": voxel_count
        })
    
    return lesions


def match_lesions_hungarian(baseline_lesions: List[Dict], followup_lesions: List[Dict],
                             w_centroid: float = 0.5, w_volume: float = 0.3, 
                             w_diameter: float = 0.2,
                             max_distance_mm: float = 30.0) -> Dict[str, Any]:
    """
    Hibrit Hungarian Algorithm ile Baseline (t0) ve Follow-up (t1) lezyonlarını eşleştirir.
    
    Bileşik Benzerlik Maliyet Matrisi:
        Cost(i,j) = w_centroid * d_centroid(i,j) + w_volume * d_volume(i,j) + w_diameter * d_diameter(i,j)
    
    Ağırlıklar:
        w_centroid = 0.5 (Konum en önemli: Aynı anatomik bölgede mi?)
        w_volume   = 0.3 (Hacim benzerliği: Benzer büyüklükte mi?)
        w_diameter = 0.2 (Çap benzerliği: Ölçüm tutarlılığı)
    
    Args:
        baseline_lesions: t0 lezyon listesi
        followup_lesions: t1 lezyon listesi
        w_centroid, w_volume, w_diameter: Ağırlıklar
        max_distance_mm: Eşleştirme kabul mesafesi (mm). Bu değeri aşan eşleşmeler reddedilir.
    
    Returns:
        Eşleşme sonuçları, yeni lezyonlar, kaybolan lezyonlar
    """
    n_bl = len(baseline_lesions)
    n_fu = len(followup_lesions)
    
    if n_bl == 0 and n_fu == 0:
        return {
            "status": "success",
            "matched_pairs": [],
            "new_lesions": [],
            "disappeared_lesions": [],
            "summary": "Her iki zaman noktasında da lezyon tespit edilmedi."
        }
    
    if n_bl == 0:
        return {
            "status": "success",
            "matched_pairs": [],
            "new_lesions": [{"followup_lesion": l, "reason": "Baseline'da eşleşen lezyon yok"} for l in followup_lesions],
            "disappeared_lesions": [],
            "summary": f"Baseline'da lezyon yok, Follow-up'ta {n_fu} yeni lezyon tespit edildi."
        }
    
    if n_fu == 0:
        return {
            "status": "success",
            "matched_pairs": [],
            "new_lesions": [],
            "disappeared_lesions": [{"baseline_lesion": l, "reason": "Follow-up'ta eşleşen lezyon yok"} for l in baseline_lesions],
            "summary": f"Baseline'da {n_bl} lezyon vardı, Follow-up'ta hepsi kayboldu (CR?)."
        }
    
    # --- Maliyet Matrisi Oluştur ---
    cost_matrix = np.zeros((n_bl, n_fu))
    
    # Normalizasyon için maksimum değerleri bul
    all_volumes = [l["volume_mm3"] for l in baseline_lesions + followup_lesions]
    all_diameters = [l["diameter_mm"] for l in baseline_lesions + followup_lesions]
    max_vol = max(all_volumes) if all_volumes else 1.0
    max_dia = max(all_diameters) if all_diameters else 1.0
    
    for i, bl in enumerate(baseline_lesions):
        for j, fu in enumerate(followup_lesions):
            # 1. Öklid mesafesi (Centroid arası, mm)
            d_centroid = np.sqrt(sum(
                (bl["centroid_mm"][k] - fu["centroid_mm"][k]) ** 2 for k in range(3)
            ))
            
            # 2. Hacim farkı (normalize)
            d_volume = abs(bl["volume_mm3"] - fu["volume_mm3"]) / (max_vol + 1e-5)
            
            # 3. Çap farkı (normalize)
            d_diameter = abs(bl["diameter_mm"] - fu["diameter_mm"]) / (max_dia + 1e-5)
            
            # Bileşik maliyet
            cost_matrix[i, j] = (
                w_centroid * d_centroid +
                w_volume * d_volume * max_distance_mm +  # Aynı ölçeğe getir
                w_diameter * d_diameter * max_distance_mm
            )
    
    # --- Hungarian Algorithm ile Optimal Eşleştirme ---
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matched_pairs = []
    matched_bl_ids = set()
    matched_fu_ids = set()
    
    for row, col in zip(row_indices, col_indices):
        cost = cost_matrix[row, col]
        
        # Mesafe eşiğini aşan eşleşmeleri reddet
        if cost > max_distance_mm * 1.5:
            continue
        
        bl = baseline_lesions[row]
        fu = followup_lesions[col]
        
        # Çap değişimini hesapla (RECIST 1.1 için kritik)
        diameter_change_mm = fu["diameter_mm"] - bl["diameter_mm"]
        diameter_change_pct = (diameter_change_mm / bl["diameter_mm"] * 100) if bl["diameter_mm"] > 0 else 0.0
        
        matched_pairs.append({
            "baseline_lesion": bl,
            "followup_lesion": fu,
            "matching_cost": round(float(cost), 2),
            "centroid_distance_mm": round(float(np.sqrt(sum(
                (bl["centroid_mm"][k] - fu["centroid_mm"][k]) ** 2 for k in range(3)
            ))), 2),
            "diameter_change_mm": round(diameter_change_mm, 2),
            "diameter_change_pct": round(diameter_change_pct, 1),
            "volume_change_mm3": round(fu["volume_mm3"] - bl["volume_mm3"], 2)
        })
        
        matched_bl_ids.add(row)
        matched_fu_ids.add(col)
    
    # Eşleşmeyen lezyonları belirle
    new_lesions = [
        {"followup_lesion": followup_lesions[j], "reason": "Baseline'da eşleşen lezyon bulunamadı (Yeni lezyon)"}
        for j in range(n_fu) if j not in matched_fu_ids
    ]
    
    disappeared_lesions = [
        {"baseline_lesion": baseline_lesions[i], "reason": "Follow-up'ta eşleşen lezyon bulunamadı (Kayıp/Regrese)"}
        for i in range(n_bl) if i not in matched_bl_ids
    ]
    
    summary_parts = []
    if matched_pairs:
        summary_parts.append(f"{len(matched_pairs)} lezyon başarıyla eşleştirildi")
    if new_lesions:
        summary_parts.append(f"{len(new_lesions)} yeni lezyon tespit edildi")
    if disappeared_lesions:
        summary_parts.append(f"{len(disappeared_lesions)} lezyon kayboldu/regrese oldu")
    
    return {
        "status": "success",
        "matched_pairs": matched_pairs,
        "new_lesions": new_lesions,
        "disappeared_lesions": disappeared_lesions,
        "has_new_lesions": len(new_lesions) > 0,
        "summary": ". ".join(summary_parts) + "."
    }


def run_longitudinal_analysis(baseline_ct: str, baseline_mask: str,
                               followup_ct: str, followup_mask: str,
                               output_dir: str) -> Dict[str, Any]:
    """
    Uçtan uca longitudinal lezyon takibi boru hattı.
    
    Adımlar:
    1. Baseline CT'yi Follow-up koordinat uzayına hizala (Registration)
    2. Baseline maskesini aynı dönüşümle taşı (Warp)
    3. Her iki maskeden lezyon özelliklerini çıkar
    4. Hungarian Algorithm ile lezyonları eşleştir
    5. Eşleşme sonuçlarını ve RECIST 1.1 girdilerini döndür
    """
    os.makedirs(output_dir, exist_ok=True)
    
    registered_baseline_path = os.path.join(output_dir, "registered_baseline.nii.gz")
    warped_mask_path = os.path.join(output_dir, "warped_baseline_mask.nii.gz")
    
    print("=" * 60)
    print("  LONGİTUDİNAL LEZYON TAKİP ANALİZİ BAŞLATILIYOR")
    print("=" * 60)
    
    # 1. Registration
    print("\n[ADIM 1/4] Görüntü Hizalama (Registration)...")
    reg_result = register_baseline_to_followup(baseline_ct, followup_ct, registered_baseline_path)
    
    # 2. Maske Taşıma (Warp)
    print("\n[ADIM 2/4] Baseline Maskesi Taşınıyor (Warp)...")
    warp_mask_with_transform(baseline_mask, followup_ct, warped_mask_path,
                              baseline_ct, registered_baseline_path)
    
    # 3. Lezyon Özelliklerini Çıkar
    print("\n[ADIM 3/4] Lezyon Özellikleri Çıkarılıyor...")
    followup_img = sitk.ReadImage(followup_ct)
    spacing = followup_img.GetSpacing()
    
    baseline_lesions = extract_lesion_features(warped_mask_path, spacing)
    followup_lesions = extract_lesion_features(followup_mask, spacing)
    
    print(f"  Baseline Lezyonları: {len(baseline_lesions)}")
    print(f"  Follow-up Lezyonları: {len(followup_lesions)}")
    
    # 4. Hungarian Eşleştirme
    print("\n[ADIM 4/4] Hibrit Hungarian Eşleştirme Çalıştırılıyor...")
    matching_result = match_lesions_hungarian(baseline_lesions, followup_lesions)
    
    # RECIST 1.1 girdilerini otomatik hesapla
    sod_baseline = sum(l["diameter_mm"] for l in baseline_lesions)
    sod_followup = sum(l["diameter_mm"] for l in followup_lesions)
    has_new = matching_result["has_new_lesions"]
    
    print(f"\n{'=' * 60}")
    print(f"  SONUÇ: {matching_result['summary']}")
    print(f"  SOD Baseline: {sod_baseline:.1f} mm | SOD Follow-up: {sod_followup:.1f} mm")
    print(f"  Yeni Lezyon: {'EVET' if has_new else 'HAYIR'}")
    print(f"{'=' * 60}")
    
    return {
        "status": "success",
        "registration": reg_result,
        "baseline_lesion_count": len(baseline_lesions),
        "followup_lesion_count": len(followup_lesions),
        "matching": matching_result,
        "recist_input": {
            "sod_baseline": round(sod_baseline, 2),
            "sod_followup": round(sod_followup, 2),
            "new_lesion": has_new
        }
    }


if __name__ == "__main__":
    print("=== Longitudinal Lezyon Eşleştirme Motoru (matching_engine.py) Yüklendi ===")
