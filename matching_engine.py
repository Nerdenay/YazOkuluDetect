"""
matching_engine.py
==================
HepaRECIST-AI - Faz 4: Boylamsal kayıt (registration) ve lezyon eşleştirme

Düzeltilmiş akış:
  1. Bazal ve takip lezyonları KENDİ doğal görüntü uzaylarında ölçülür
     (çap / hacim deformasyondan etkilenmez).
  2. Rigid + B-Spline kayıt yalnızca KONUM eşleştirmesi için kullanılır:
     bazal lezyon kimlik haritası (her bileşen kendi numarasıyla) takip uzayına
     taşınır ve her bazal lezyonun takip uzayındaki merkezi bulunur.
  3. Kayıt kalitesi organ maskelerinin örtüşmesiyle (Dice) kontrol edilir.
  4. Macar algoritması ile eşleştirme.
  5. Eşleşmeyen takip lezyonları "aday yeni lezyon" olarak işaretlenir ve
     bazal görüntüde aynı bölgeye geri dönülerek kontrol edilir. Hiçbir aday
     otomatik olarak PD'ye çevrilmez.
  6. Hedef lezyonlar YALNIZCA bazalda seçilir (organ başına <= 2, toplam <= 5)
     ve takipte aynı lezyonlar izlenir.
"""

import os
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from lesion_geometry import extract_lesions, select_target_lesions

# Kayıt kalite kontrolü için kullanılan organlar ve Dice eşiği
QC_ORGANS = {1: "karaciger", 2: "dalak", 3: "bobrek"}
QC_MIN_DICE = 0.70


# ── 1. KAYIT (REGISTRATION) ──────────────────────────────────────────────────

def register_baseline_to_followup(baseline_nifti: str, followup_nifti: str,
                                  output_registered_path: Optional[str] = None
                                  ) -> Tuple[sitk.Transform, Dict[str, Any]]:
    """
    Bazal (t0) BT'yi takip (t1) uzayına hizalar. Dönen dönüşüm, takip
    uzayındaki bir noktayı bazal uzaydaki karşılığına eşler (sitk.Resample yönü).
    """
    if not os.path.exists(baseline_nifti) or not os.path.exists(followup_nifti):
        raise FileNotFoundError("BT dosyaları bulunamadı.")

    fixed = sitk.ReadImage(followup_nifti, sitk.sitkFloat32)
    moving = sitk.ReadImage(baseline_nifti, sitk.sitkFloat32)
    # Masa, hava gibi yapıların MI metriğini domine etmemesi için HU kırpma
    fixed_c = sitk.Clamp(fixed, sitk.sitkFloat32, -200.0, 400.0)
    moving_c = sitk.Clamp(moving, sitk.sitkFloat32, -200.0, 400.0)

    init = sitk.CenteredTransformInitializer(
        fixed_c, moving_c, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid_reg.SetMetricSamplingStrategy(rigid_reg.RANDOM)
    rigid_reg.SetMetricSamplingPercentage(0.02, seed=42)
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=150,
                                            convergenceMinimumValue=1e-6)
    rigid_reg.SetOptimizerScalesFromPhysicalShift()
    rigid_reg.SetInitialTransform(init, inPlace=False)
    rigid_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    rigid_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    rigid_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rigid = rigid_reg.Execute(fixed_c, moving_c)

    moving_rigid = sitk.Resample(moving_c, fixed_c, rigid, sitk.sitkLinear, -200.0, sitk.sitkFloat32)

    grid_spacing = 80.0
    phys = [fixed.GetSize()[i] * fixed.GetSpacing()[i] for i in range(3)]
    mesh = [max(2, int(round(p / grid_spacing))) for p in phys]
    bspline = sitk.BSplineTransformInitializer(fixed_c, mesh, order=3)
    bs_reg = sitk.ImageRegistrationMethod()
    bs_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    bs_reg.SetMetricSamplingStrategy(bs_reg.RANDOM)
    bs_reg.SetMetricSamplingPercentage(0.03, seed=42)
    bs_reg.SetInterpolator(sitk.sitkLinear)
    bs_reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-4, numberOfIterations=40,
                                costFunctionConvergenceFactor=1e+7)
    bs_reg.SetInitialTransform(bspline, inPlace=False)
    bs_reg.SetShrinkFactorsPerLevel([2])
    bs_reg.SetSmoothingSigmasPerLevel([1])
    bs_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    final_bspline = bs_reg.Execute(fixed_c, moving_rigid)

    # Resample sırası: önce B-spline (sabit uzayda), sonra rigid -> bazal uzay.
    # CompositeTransform son eklenen dönüşümü ilk uygular.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(rigid)
    composite.AddTransform(final_bspline)

    if output_registered_path:
        reg_ct = sitk.Resample(moving, fixed, composite, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
        sitk.WriteImage(reg_ct, output_registered_path)

    metrics = {
        "rigid_metric": round(float(rigid_reg.GetMetricValue()), 4),
        "bspline_metric": round(float(bs_reg.GetMetricValue()), 4),
    }
    return composite, metrics


def warp_label_image(label_img: sitk.Image, reference_img: sitk.Image,
                     transform: sitk.Transform) -> sitk.Image:
    """Etiket görüntüsünü (en yakın komşu) referans uzayına taşır."""
    return sitk.Resample(label_img, reference_img, transform,
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32)


def warp_baseline_mask(baseline_mask_path: str, followup_nifti: str,
                       composite_transform: sitk.Transform,
                       output_warped_mask_path: str) -> str:
    """Geriye dönük uyumluluk: bazal maskeyi takip uzayına taşıyıp kaydeder."""
    ref = sitk.ReadImage(followup_nifti)
    mask = sitk.ReadImage(baseline_mask_path, sitk.sitkUInt8)
    warped = sitk.Cast(warp_label_image(mask, ref, composite_transform), sitk.sitkUInt8)
    sitk.WriteImage(warped, output_warped_mask_path)
    return output_warped_mask_path


# ── 2. KAYIT KALİTE KONTROLÜ ─────────────────────────────────────────────────

def registration_qc(warped_bl_labels: np.ndarray, fu_labels: np.ndarray,
                    min_dice: float = QC_MIN_DICE) -> Dict[str, Any]:
    """Taşınmış bazal organ maskesi ile takip organ maskesinin Dice örtüşmesi."""
    dices = {}
    for lab, name in QC_ORGANS.items():
        a = warped_bl_labels == lab
        b = fu_labels == lab
        sa, sb = int(a.sum()), int(b.sum())
        if sa == 0 or sb == 0:
            continue
        dices[name] = round(2.0 * float((a & b).sum()) / (sa + sb), 3)

    if not dices:
        return {"organ_dice": {}, "reliable": None,
                "message": "Organ maskesi bulunamadığından kayıt kalitesi ölçülemedi; eşleştirme radyologca kontrol edilmelidir."}
    worst = min(dices.values())
    reliable = worst >= min_dice
    msg = ("Kayıt kalitesi yeterli." if reliable else
           f"Kayıt kalitesi düşük (en düşük organ Dice {worst:.2f} < {min_dice:.2f}). "
           "Eşleştirme, yeni ve kaybolan lezyon sonuçları güvenilmez olabilir.")
    return {"organ_dice": dices, "min_dice": worst, "reliable": reliable, "message": msg}


# ── 3. MACAR EŞLEŞTİRME ──────────────────────────────────────────────────────

def match_lesions_hungarian(baseline_lesions: List[Dict], followup_lesions: List[Dict],
                            w_centroid: float = 0.6, w_volume: float = 0.2,
                            w_diameter: float = 0.2, max_distance_mm: float = 25.0) -> Dict[str, Any]:
    """
    Bazal ve takip lezyonlarını minimum maliyetle eşler.
    Konum için bazal lezyonların takip uzayındaki merkezi ('centroid_fu_mm',
    yoksa 'centroid_mm'), boyut için doğal uzay ölçümleri kullanılır.
    """
    n_bl, n_fu = len(baseline_lesions), len(followup_lesions)
    empty = {"matched_pairs": [], "unmatched_followup": [], "unmatched_baseline": []}
    if n_bl == 0 or n_fu == 0:
        empty["unmatched_followup"] = list(range(n_fu))
        empty["unmatched_baseline"] = list(range(n_bl))
        return empty

    def pos(l):
        return np.asarray(l.get("centroid_fu_mm") or l["centroid_mm"], dtype=float)

    all_l = baseline_lesions + followup_lesions
    max_vol = max(l["volume_mm3"] for l in all_l) or 1.0
    max_dia = max(l["diameter_mm"] for l in all_l) or 1.0

    cost = np.full((n_bl, n_fu), 1e5)
    dist = np.zeros((n_bl, n_fu))
    for i, bl in enumerate(baseline_lesions):
        for j, fu in enumerate(followup_lesions):
            d = float(np.linalg.norm(pos(bl) - np.asarray(fu["centroid_mm"], dtype=float)))
            dist[i, j] = d
            if d <= max_distance_mm:
                cost[i, j] = (w_centroid * d
                              + w_volume * abs(bl["volume_mm3"] - fu["volume_mm3"]) / max_vol * max_distance_mm
                              + w_diameter * abs(bl["diameter_mm"] - fu["diameter_mm"]) / max_dia * max_distance_mm)

    rows, cols = linear_sum_assignment(cost)
    pairs, mb, mf = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] >= 1e4:
            continue
        pairs.append((int(r), int(c), round(float(dist[r, c]), 2)))
        mb.add(int(r))
        mf.add(int(c))
    return {
        "matched_pairs": pairs,
        "unmatched_followup": [j for j in range(n_fu) if j not in mf],
        "unmatched_baseline": [i for i in range(n_bl) if i not in mb],
    }


# ── 4. YENİ LEZYON ADAYI DOĞRULAMA ───────────────────────────────────────────

def _check_new_lesion_candidate(fu_lesion: Dict, fu_components: np.ndarray,
                                warped_bl_lesion_bin: np.ndarray,
                                bl_ct: sitk.Image, transform: sitk.Transform,
                                fu_ct_arr: np.ndarray, min_size_mm: float,
                                slice_thickness_mm: Optional[float],
                                hu_similarity: float = 15.0,
                                overlap_margin_vox: int = 3) -> Dict[str, Any]:
    """
    Eşleşmeyen bir takip lezyonunu sınıflandırır:
      - too_small                : boyut eşiğinin altında (raporlanır, karar dışı)
      - present_at_baseline_mask : bazal maskede aynı bölgede lezyon var (eşleştirme hatası olabilir)
      - possibly_missed_at_baseline : bazal görüntüde aynı yerde benzer dansite var
      - candidate_new            : aday yeni lezyon (radyolog onayı gerekli)
    """
    info = {"followup_lesion": fu_lesion, "requires_review": True}

    z_ok = slice_thickness_mm is None or fu_lesion["z_extent_mm"] >= float(slice_thickness_mm)
    if fu_lesion["diameter_mm"] < min_size_mm or not z_ok:
        info.update(status="too_small", requires_review=False,
                    reason=f"Boyut eşiğinin altında (< {min_size_mm:.0f} mm veya tek kesitte).")
        return info

    region = fu_components == fu_lesion["id"]
    near = ndimage.binary_dilation(region, iterations=overlap_margin_vox)
    if (near & warped_bl_lesion_bin).any():
        info.update(status="present_at_baseline_mask",
                    reason="Bazal segmentasyonda aynı bölgede lezyon mevcut; yeni değil, eşleştirme kontrol edilmeli.")
        return info

    # Takip merkezini bazal uzaya taşı ve aynı bölgede dansiteyi karşılaştır
    try:
        p_bl = transform.TransformPoint(tuple(float(c) for c in fu_lesion["centroid_mm"]))
        idx = bl_ct.TransformPhysicalPointToContinuousIndex(p_bl)
        bl_arr = sitk.GetArrayViewFromImage(bl_ct)
        sp = bl_ct.GetSpacing()
        r_mm = max(2.0, fu_lesion["diameter_mm"] / 2.0)
        rx, ry, rz = [max(1, int(round(r_mm / s))) for s in sp]
        cx, cy, cz = [int(round(v)) for v in idx]
        zs, ys, xs = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
        ball = (xs * sp[0]) ** 2 + (ys * sp[1]) ** 2 + (zs * sp[2]) ** 2 <= r_mm ** 2
        z0, y0, x0 = cz - rz, cy - ry, cx - rx
        if z0 < 0 or y0 < 0 or x0 < 0 or cz + rz >= bl_arr.shape[0] or \
           cy + ry >= bl_arr.shape[1] or cx + rx >= bl_arr.shape[2]:
            raise IndexError("Bazal görüntü sınırı dışında")
        bl_patch = bl_arr[z0:cz + rz + 1, y0:cy + ry + 1, x0:cx + rx + 1][ball]
        bl_hu = float(bl_patch.mean())
        fu_hu = float(fu_ct_arr[region].mean())
        info["baseline_hu_at_location"] = round(bl_hu, 1)
        info["followup_hu"] = round(fu_hu, 1)
        if abs(bl_hu - fu_hu) <= hu_similarity:
            info.update(status="possibly_missed_at_baseline",
                        reason=(f"Bazalda aynı bölgede benzer dansite ({bl_hu:.0f} vs {fu_hu:.0f} HU); "
                                "lezyon bazalda mevcut olup segmente edilmemiş olabilir."))
            return info
    except Exception as exc:  # noqa: BLE001
        info["baseline_check_error"] = str(exc)

    info.update(status="candidate_new", reason="Aday yeni lezyon — radyolog onayı gerekli.")
    return info


# ── 5. ANA ÇALIŞTIRICI ───────────────────────────────────────────────────────

def run_longitudinal_analysis(baseline_ct: str, baseline_mask: str,
                              followup_ct: str, followup_mask: str,
                              output_dir: str, lesion_label_id: int = 8,
                              baseline_slice_thickness_mm: Optional[float] = None,
                              followup_slice_thickness_mm: Optional[float] = None,
                              min_new_lesion_mm: float = 5.0,
                              max_distance_mm: float = 25.0) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    reg_ct_path = os.path.join(output_dir, "registered_baseline_ct.nii.gz")
    warped_mask_path = os.path.join(output_dir, "warped_baseline_mask.nii.gz")

    bl_ct_img = sitk.ReadImage(baseline_ct, sitk.sitkFloat32)
    fu_ct_img = sitk.ReadImage(followup_ct, sitk.sitkFloat32)
    bl_mask_img = sitk.ReadImage(baseline_mask, sitk.sitkUInt8)
    fu_mask_img = sitk.ReadImage(followup_mask, sitk.sitkUInt8)

    # 1. Doğal uzayda ölçüm
    print("[1/5] Lezyonlar doğal uzaylarında ölçülüyor...")
    bl_lesions, bl_comp = extract_lesions(bl_mask_img, lesion_label_id,
                                          baseline_slice_thickness_mm, bl_ct_img)
    fu_lesions, fu_comp = extract_lesions(fu_mask_img, lesion_label_id,
                                          followup_slice_thickness_mm, fu_ct_img)

    # 2. Kayıt
    print("[2/5] Rigid + B-Spline kayıt hesaplanıyor...")
    transform, reg_metrics = register_baseline_to_followup(baseline_ct, followup_ct, reg_ct_path)

    # Bazal organ maskesi (QC + yeni lezyon kontrolü) ve lezyon kimlik haritası taşınır
    warped_bl = sitk.GetArrayFromImage(warp_label_image(bl_mask_img, fu_mask_img, transform))
    sitk.WriteImage(sitk.Cast(warp_label_image(bl_mask_img, fu_mask_img, transform), sitk.sitkUInt8),
                    warped_mask_path)
    inst_img = sitk.GetImageFromArray(bl_comp.astype(np.uint32))
    inst_img.CopyInformation(bl_mask_img)
    warped_inst = sitk.GetArrayFromImage(warp_label_image(inst_img, fu_mask_img, transform))

    for les in bl_lesions:
        vox = warped_inst == les["id"]
        if vox.any():
            com = ndimage.center_of_mass(vox)
            p = fu_mask_img.TransformContinuousIndexToPhysicalPoint([com[2], com[1], com[0]])
            les["centroid_fu_mm"] = [round(float(c), 2) for c in p]
        else:
            les["centroid_fu_mm"] = None  # Çok küçük lezyon taşımada kayboldu; doğal merkez kullanılır

    # 3. Kayıt kalite kontrolü
    print("[3/5] Kayıt kalitesi (organ Dice) kontrol ediliyor...")
    fu_labels = sitk.GetArrayFromImage(fu_mask_img)
    qc = registration_qc(warped_bl, fu_labels)
    print(f"      {qc['message']}")

    # 4. Eşleştirme
    print("[4/5] Macar algoritması ile eşleştirme...")
    m = match_lesions_hungarian(bl_lesions, fu_lesions, max_distance_mm=max_distance_mm)

    matched_pairs = []
    for i, j, d in m["matched_pairs"]:
        bl, fu = bl_lesions[i], fu_lesions[j]
        ch = fu["diameter_mm"] - bl["diameter_mm"]
        matched_pairs.append({
            "baseline_id": bl["id"], "followup_id": fu["id"], "organ": bl["organ"],
            "baseline_diameter": bl["diameter_mm"], "followup_diameter": fu["diameter_mm"],
            "diameter_change_mm": round(ch, 2),
            "diameter_change_pct": round(ch / bl["diameter_mm"] * 100, 1) if bl["diameter_mm"] > 0 else 0.0,
            "centroid_distance_mm": d,
        })
    fu_by_bl = {p["baseline_id"]: p for p in matched_pairs}

    # 5. Yeni / kaybolan lezyonlar
    print("[5/5] Yeni ve kaybolan lezyonlar değerlendiriliyor...")
    warped_bl_lesion_bin = warped_bl == lesion_label_id
    fu_ct_arr = sitk.GetArrayFromImage(fu_ct_img)
    new_checks = [
        _check_new_lesion_candidate(fu_lesions[j], fu_comp, warped_bl_lesion_bin, bl_ct_img,
                                    transform, fu_ct_arr, min_new_lesion_mm,
                                    followup_slice_thickness_mm)
        for j in m["unmatched_followup"]
    ]
    candidates = [c for c in new_checks if c["status"] == "candidate_new"]
    disappeared = [{"baseline_lesion": bl_lesions[i], "requires_review": True,
                    "reason": "Takipte eşleşen lezyon yok — kaybolmuş olabilir; radyolog doğrulamalı."}
                   for i in m["unmatched_baseline"]]

    # 6. Hedef lezyonlar: yalnızca bazalda seçilir, takipte aynıları izlenir
    targets = select_target_lesions(bl_lesions)
    target_rows = []
    for t in targets:
        pair = fu_by_bl.get(t["id"])
        target_rows.append({
            "baseline_id": t["id"], "organ": t["organ"],
            "baseline_diameter": t["diameter_mm"],
            "followup_diameter": pair["followup_diameter"] if pair else 0.0,
            "status": "matched" if pair else "not_found_at_followup",
        })
    sod_bl = sum(r["baseline_diameter"] for r in target_rows)
    sod_fu = sum(r["followup_diameter"] for r in target_rows)
    sod_change_pct = round((sod_fu - sod_bl) / sod_bl * 100, 2) if sod_bl > 0 else 0.0

    requires_review = bool(candidates or disappeared or any(
        c["status"] in ("present_at_baseline_mask", "possibly_missed_at_baseline") for c in new_checks)
        or qc.get("reliable") is not True)

    return {
        "status": "success",
        "registration_metrics": reg_metrics,
        "registration_qc": qc,
        "baseline_lesions": bl_lesions,
        "followup_lesions_detail": fu_lesions,
        "target_lesions": target_rows,
        "baseline_targets": len(targets),
        "followup_lesions": len(fu_lesions),
        "matching": {
            "status": "success",
            "matched_pairs": matched_pairs,
            "new_lesions": candidates,              # yalnızca aday yeni lezyonlar
            "unmatched_followup_review": new_checks,  # tüm eşleşmeyenler + durumları
            "disappeared_lesions": disappeared,
            "has_new_lesions": False,               # onaylanmış yeni lezyon (radyolog)
            "new_lesion_candidates": len(candidates),
        },
        "requires_radiologist_review": requires_review,
        "recist_metrics": {
            "sod_baseline": round(sod_bl, 2),
            "sod_followup": round(sod_fu, 2),
            "sod_change_pct": sod_change_pct,
            "has_new_lesions": False,
            "new_lesion_candidates": len(candidates),
        },
        "recist_input": {
            "sod_baseline": round(sod_bl, 2),
            "sod_followup": round(sod_fu, 2),
            "new_lesion": False,
            "new_lesion_candidates": len(candidates),
        },
    }
