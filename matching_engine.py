"""
matching_engine.py
==================
HepaRECIST-AI - Faz 4: 4D Longitudinal Registration & Hungarian Lezyon Eşleştirme Motoru
- Rigid (Euler3D) + Deformable B-Spline Registration (SimpleITK)
- Baseline lezyon maskesini hesaplanan dönüşümle deforme etme (Nearest Neighbor Warp)
- Fiziksel uzayda (LPS) kütle merkezi ve aksiyel düzlemde RECIST 1.1 en uzun çap hesabı
- Hibrit maliyet matrisi üzerinden Macar (Hungarian) Algoritması ile eşleştirme
- RECIST 1.1 Hedef Lezyon (Target Lesion: >=10mm, max 5 lezyon) ve SOD analizi
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Tuple


# ── 1. REGISTRATION & MASK WARPING ──────────────────────────────────────────

def register_baseline_to_followup(baseline_nifti: str, followup_nifti: str, 
                                   output_registered_path: str) -> Tuple[sitk.CompositeTransform, Dict[str, Any]]:
    """
    Baseline (t0) CT'yi Follow-up (t1) referans uzayına hizalar.
    Hesaplanan bileşik dönüşümü (Rigid + B-Spline) döndürür.
    """
    if not os.path.exists(baseline_nifti) or not os.path.exists(followup_nifti):
        raise FileNotFoundError("CT dosyaları bulunamadı.")

    fixed_image = sitk.ReadImage(followup_nifti, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(baseline_nifti, sitk.sitkFloat32)

    # Aşama 1: Rigid Registration
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid_reg.SetMetricSamplingStrategy(rigid_reg.RANDOM)
    rigid_reg.SetMetricSamplingPercentage(0.01)
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=150, convergenceMinimumValue=1e-6)
    rigid_reg.SetOptimizerScalesFromPhysicalShift()
    rigid_reg.SetInitialTransform(initial_transform, inPlace=False)
    rigid_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    rigid_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    rigid_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    rigid_transform = rigid_reg.Execute(fixed_image, moving_image)

    # Aşama 2: Deformable B-Spline Registration
    moving_resampled = sitk.Resample(moving_image, fixed_image, rigid_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    grid_spacing = [80.0, 80.0, 80.0]
    physical_size = [fixed_image.GetSize()[i] * fixed_image.GetSpacing()[i] for i in range(3)]
    mesh_size = [max(2, int(round(physical_size[i] / grid_spacing[i]))) for i in range(3)]

    bspline_transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size, order=3)
    bspline_reg = sitk.ImageRegistrationMethod()
    bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
    bspline_reg.SetMetricSamplingPercentage(0.03)
    bspline_reg.SetInterpolator(sitk.sitkLinear)
    bspline_reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-4, numberOfIterations=40, costFunctionConvergenceFactor=1e+7)
    bspline_reg.SetInitialTransform(bspline_transform, inPlace=False)
    bspline_reg.SetShrinkFactorsPerLevel([2])
    bspline_reg.SetSmoothingSigmasPerLevel([1])
    bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_bspline = bspline_reg.Execute(fixed_image, moving_resampled)

    # Bileşik Dönüşüm (Rigid + B-Spline)
    composite_transform = sitk.CompositeTransform(3)
    composite_transform.AddTransform(rigid_transform)
    composite_transform.AddTransform(final_bspline)

    # Görüntüyü yeniden örnekle ve kaydet
    registered_ct = sitk.Resample(moving_image, fixed_image, composite_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    sitk.WriteImage(registered_ct, output_registered_path)

    metrics = {
        "rigid_metric": round(float(rigid_reg.GetMetricValue()), 4),
        "bspline_metric": round(float(bspline_reg.GetMetricValue()), 4)
    }
    return composite_transform, metrics


def warp_baseline_mask(baseline_mask_path: str, followup_nifti: str,
                       composite_transform: sitk.CompositeTransform,
                       output_warped_mask_path: str) -> str:
    """
    Baseline maskesini hesaplanan CompositeTransform ile Nearest Neighbor kullanarak
    follow-up koordinat uzayına taşır.
    """
    fixed_image = sitk.ReadImage(followup_nifti, sitk.sitkFloat32)
    baseline_mask = sitk.ReadImage(baseline_mask_path, sitk.sitkUInt8)

    warped_mask = sitk.Resample(
        baseline_mask, fixed_image, composite_transform,
        sitk.sitkNearestNeighbor, 0, baseline_mask.GetPixelID()
    )
    sitk.WriteImage(warped_mask, output_warped_mask_path)
    return output_warped_mask_path


# ── 2. RECIST 1.1 LEZYON ÖZELLİKLERİ VE ÇAP HESABI ──────────────────────────

def calculate_axial_recist_diameter(region_voxels_3d: np.ndarray, spacing: Tuple[float, ...]) -> float:
    """
    RECIST 1.1 Standardı: Aksiyel (X-Y) düzlemdeki maksimum 2D Feret çapını hesaplar.
    """
    max_diameter = 0.0
    z_indices = np.unique(np.where(region_voxels_3d)[0])

    for z in z_indices:
        slice_2d = region_voxels_3d[z, :, :]
        y_coords, x_coords = np.where(slice_2d)
        if len(x_coords) < 3:
            continue

        pts = np.column_stack((x_coords * spacing[0], y_coords * spacing[1]))
        if len(pts) > 100:
            step = max(1, len(pts) // 50)
            pts = pts[::step]

        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        max_in_slice = float(np.max(dists))
        if max_in_slice > max_diameter:
            max_diameter = max_in_slice

    return round(max_diameter, 2)


def extract_lesion_features(mask_path: str, lesion_label_id: int = 8) -> List[Dict[str, Any]]:
    """
    Maskeden lezyon bölgelerini ayıklar (varsayılan Label 8, yoksa 2 veya 1 fallback).
    Kütle merkezini fiziksel koordinat uzayında (mm), çapı ise RECIST 1.1 aksiyel düzleminde çıkarır.
    """
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)  # Shape: (Z, Y, X)
    spacing = mask_img.GetSpacing()              # (sx, sy, sz)
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    # Eğer belirtilen etiket maskede yoksa lezyon tespit edilmemiştir
    unique_vals = set(np.unique(mask_arr))
    if lesion_label_id not in unique_vals:
        return []

    binary_lesions = (mask_arr == lesion_label_id)
    labeled_mask, num_features = ndimage.label(binary_lesions)

    lesions = []
    for region_id in range(1, num_features + 1):
        region_voxels = (labeled_mask == region_id)
        voxel_count = int(np.sum(region_voxels))

        if voxel_count < 15:  # Çok küçük gürültüleri filtrele
            continue

        centroid_zyx = ndimage.center_of_mass(region_voxels)
        continuous_idx = [centroid_zyx[2], centroid_zyx[1], centroid_zyx[0]]
        physical_point = mask_img.TransformContinuousIndexToPhysicalPoint(continuous_idx)

        diameter_mm = calculate_axial_recist_diameter(region_voxels, spacing)
        volume_mm3 = round(float(voxel_count * voxel_vol), 2)

        lesions.append({
            "id": region_id,
            "centroid_mm": physical_point,
            "volume_mm3": volume_mm3,
            "diameter_mm": diameter_mm,
            "is_target": diameter_mm >= 10.0,
            "voxel_count": voxel_count
        })

    return lesions


# ── 3. HİBRİT MACAR (HUNGARIAN) EŞLEŞTİRME ────────────────────────────────────

def match_lesions_hungarian(baseline_lesions: List[Dict], followup_lesions: List[Dict],
                             w_centroid: float = 0.6, w_volume: float = 0.2, 
                             w_diameter: float = 0.2, max_distance_mm: float = 25.0) -> Dict[str, Any]:
    """
    t0 ve t1 lezyonlarını minimum maliyetli bipartite eşleştirme ile bağlar.
    """
    n_bl = len(baseline_lesions)
    n_fu = len(followup_lesions)

    if n_bl == 0 and n_fu == 0:
        return {"status": "success", "matched_pairs": [], "new_lesions": [], "disappeared_lesions": [], "has_new_lesions": False}

    if n_bl == 0:
        return {
            "status": "success", "matched_pairs": [],
            "new_lesions": [{"followup_lesion": l, "reason": "Baseline'da lezyon yok"} for l in followup_lesions],
            "disappeared_lesions": [], "has_new_lesions": True
        }

    if n_fu == 0:
        return {
            "status": "success", "matched_pairs": [], "new_lesions": [],
            "disappeared_lesions": [{"baseline_lesion": l, "reason": "Lezyon regrese oldu"} for l in baseline_lesions],
            "has_new_lesions": False
        }

    cost_matrix = np.zeros((n_bl, n_fu))
    max_vol = max([l["volume_mm3"] for l in baseline_lesions + followup_lesions]) or 1.0
    max_dia = max([l["diameter_mm"] for l in baseline_lesions + followup_lesions]) or 1.0

    for i, bl in enumerate(baseline_lesions):
        for j, fu in enumerate(followup_lesions):
            d_centroid = np.sqrt(sum((bl["centroid_mm"][k] - fu["centroid_mm"][k]) ** 2 for k in range(3)))
            d_volume = abs(bl["volume_mm3"] - fu["volume_mm3"]) / max_vol
            d_diameter = abs(bl["diameter_mm"] - fu["diameter_mm"]) / max_dia

            if d_centroid > max_distance_mm:
                cost_matrix[i, j] = 1e5
            else:
                cost_matrix[i, j] = (w_centroid * d_centroid) + (w_volume * d_volume * max_distance_mm) + (w_diameter * d_diameter * max_distance_mm)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_pairs, matched_bl, matched_fu = [], set(), set()
    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] >= 1e4:
            continue

        bl = baseline_lesions[r]
        fu = followup_lesions[c]
        d_change = fu["diameter_mm"] - bl["diameter_mm"]
        d_pct = round((d_change / bl["diameter_mm"] * 100), 1) if bl["diameter_mm"] > 0 else 0.0

        matched_pairs.append({
            "baseline_id": bl["id"], "followup_id": fu["id"],
            "baseline_diameter": bl["diameter_mm"], "followup_diameter": fu["diameter_mm"],
            "diameter_change_mm": round(d_change, 2), "diameter_change_pct": d_pct,
            "centroid_distance_mm": round(float(np.sqrt(sum((bl["centroid_mm"][k] - fu["centroid_mm"][k]) ** 2 for k in range(3)))), 2)
        })
        matched_bl.add(r)
        matched_fu.add(c)

    new_lesions = [followup_lesions[j] for j in range(n_fu) if j not in matched_fu]
    disappeared_lesions = [baseline_lesions[i] for i in range(n_bl) if i not in matched_bl]

    return {
        "status": "success",
        "matched_pairs": matched_pairs,
        "new_lesions": new_lesions,
        "disappeared_lesions": disappeared_lesions,
        "has_new_lesions": len(new_lesions) > 0
    }


# ── 4. ANA ÇALIŞTIRICI FONKSİYON ─────────────────────────────────────────────

def run_longitudinal_analysis(baseline_ct: str, baseline_mask: str,
                               followup_ct: str, followup_mask: str,
                               output_dir: str, lesion_label_id: int = 8) -> Dict[str, Any]:
    """
    Longitudinal takip sürecini eksiksiz yürütür ve RECIST 1.1 SOD değerlerini döndürür.
    """
    os.makedirs(output_dir, exist_ok=True)
    reg_ct_path = os.path.join(output_dir, "registered_baseline_ct.nii.gz")
    warped_mask_path = os.path.join(output_dir, "warped_baseline_mask.nii.gz")

    # 1. Registration
    print("[1/4] B-Spline Registration hesaplanıyor...")
    composite_transform, reg_metrics = register_baseline_to_followup(baseline_ct, followup_ct, reg_ct_path)

    # 2. Maskeyi Dönüşümle Taşı (Nearest Neighbor)
    print("[2/4] Baseline maskesi transform ile taşınıyor...")
    warp_baseline_mask(baseline_mask, followup_ct, composite_transform, warped_mask_path)

    # 3. Özellik Çıkarımı (Label 8: Malign Lezyonlar)
    print(f"[3/4] Lezyonlar (Label {lesion_label_id}) ve RECIST çapları çıkarılıyor...")
    bl_lesions = extract_lesion_features(warped_mask_path, lesion_label_id=lesion_label_id)
    fu_lesions = extract_lesion_features(followup_mask, lesion_label_id=lesion_label_id)

    # 4. Eşleştirme
    print("[4/4] Hungarian eşleştirme uygulanıyor...")
    match_res = match_lesions_hungarian(bl_lesions, fu_lesions)

    # RECIST 1.1: Hedef lezyonları filtrele (>=10mm, en büyük max 5 tanesi)
    target_bl = sorted([l for l in bl_lesions if l["is_target"]], key=lambda x: x["diameter_mm"], reverse=True)[:5]
    sod_bl = sum(l["diameter_mm"] for l in target_bl)

    # Eşleşen target lezyonların follow-up toplamı
    target_bl_ids = {l["id"] for l in target_bl}
    sod_fu = 0.0
    for pair in match_res["matched_pairs"]:
        if pair["baseline_id"] in target_bl_ids:
            sod_fu += pair["followup_diameter"]

    sod_change_pct = round(((sod_fu - sod_bl) / sod_bl * 100), 2) if sod_bl > 0 else 0.0

    return {
        "status": "success",
        "registration_metrics": reg_metrics,
        "baseline_targets": len(target_bl),
        "followup_lesions": len(fu_lesions),
        "matching": match_res,
        "recist_metrics": {
            "sod_baseline": round(sod_bl, 2),
            "sod_followup": round(sod_fu, 2),
            "sod_change_pct": sod_change_pct,
            "has_new_lesions": match_res["has_new_lesions"]
        },
        "recist_input": {
            "sod_baseline": round(sod_bl, 2),
            "sod_followup": round(sod_fu, 2),
            "new_lesion": match_res["has_new_lesions"]
        }
    }
