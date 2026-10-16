"""
lesion_geometry.py
==================
HepaRECIST-AI - Ortak lezyon ölçüm modülü.

Tüm modüller (inference, matching_engine, radiomics) lezyon çıkarımı ve
çap ölçümü için bu modülü kullanır; böylece aynı lezyon her yerde aynı
şekilde ölçülür.

- En uzun aksiyel çap: her kesitte lezyon konturunun konveks zarfı üzerindeki
  TÜM nokta çiftleri arasındaki en büyük mesafe (seyreltme yok, kenar noktası
  atlanmaz). Piksel merkezleri arası mesafeye bir piksel boyu eklenir
  (kenardan kenara ölçüm, radyoloğun kaliperine yakın).
- Organ ataması: lezyonun çevresindeki (dilate kabuk) en sık organ etiketi.
- RECIST 1.1 ölçülebilirlik: en uzun çap >= max(10 mm, 2 x orijinal kesit kalınlığı).
"""

from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

# nnU-Net etiket şeması (dataset.json ile aynı)
ORGAN_NAMES = {
    0: "diger",
    1: "karaciger",
    2: "dalak",
    3: "bobrek",
    4: "pankreas",
    5: "safra_kesesi",
    6: "mide",
    7: "aorta",
}

MIN_COMPONENT_VOXELS = 15


def max_axial_diameter(region_zyx: np.ndarray, spacing_xyz: Tuple[float, ...]) -> float:
    """
    RECIST 1.1 en uzun aksiyel çap (mm).

    region_zyx : (Z, Y, X) boolean dizi (SimpleITK GetArrayFromImage düzeni)
    spacing_xyz: SimpleITK GetSpacing() çıktısı (sx, sy, sz)
    """
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    best = 0.0
    for z in np.unique(np.nonzero(region_zyx)[0]):
        sl = region_zyx[z]
        # Yalnızca kontur pikselleri (iç pikseller en uzun çapa katkı vermez)
        contour = sl & ~ndimage.binary_erosion(sl)
        ys, xs = np.nonzero(contour)
        if len(xs) == 0:
            continue
        pts = np.column_stack((xs * sx, ys * sy))
        if len(pts) >= 3:
            try:
                pts = pts[ConvexHull(pts).vertices]
            except QhullError:
                pass  # Doğrusal dizilim: tüm noktalarla devam
        if len(pts) >= 2:
            d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).max()
        else:
            d = 0.0
        # Merkezden merkeze -> kenardan kenara (yaklaşık bir piksel)
        d += min(sx, sy)
        best = max(best, float(d))
    return round(best, 2)


def assign_organ(region_zyx: np.ndarray, label_arr: np.ndarray, lesion_label_id: int,
                 shell_voxels: int = 3) -> str:
    """Lezyonun çevresindeki kabukta en sık görülen organ etiketini döndürür."""
    dil = ndimage.binary_dilation(region_zyx, iterations=shell_voxels)
    shell = dil & ~region_zyx
    vals = label_arr[shell]
    vals = vals[(vals != lesion_label_id)]
    organ_vals = vals[vals > 0]
    if organ_vals.size == 0:
        return ORGAN_NAMES[0]
    counts = np.bincount(organ_vals.astype(np.int64))
    return ORGAN_NAMES.get(int(np.argmax(counts)), ORGAN_NAMES[0])


def measurability_threshold_mm(slice_thickness_mm: Optional[float]) -> float:
    """RECIST 1.1: >= 10 mm ve >= 2 x kesit kalınlığı."""
    if slice_thickness_mm is None or slice_thickness_mm <= 0:
        return 10.0
    return max(10.0, 2.0 * float(slice_thickness_mm))


def extract_lesions(mask_img: sitk.Image, lesion_label_id: int = 8,
                    slice_thickness_mm: Optional[float] = None,
                    ct_img: Optional[sitk.Image] = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Maskeden bağımsız lezyon odaklarını çıkarır.

    Dönüş: (lezyon listesi, bağlı bileşen etiket dizisi (Z,Y,X))
    Her lezyonun 'id' alanı bu etiket dizisindeki bileşen numarasıdır.
    """
    label_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    thr = measurability_threshold_mm(slice_thickness_mm)
    ct_arr = sitk.GetArrayFromImage(ct_img) if ct_img is not None else None

    binary = label_arr == lesion_label_id
    components, n = ndimage.label(binary)
    if n == 0:
        return [], components

    lesions = []
    objects = ndimage.find_objects(components)
    for cid in range(1, n + 1):
        sl = objects[cid - 1]
        if sl is None:
            continue
        # Hız için bileşenin sınırlayıcı kutusu (+ organ kabuğu için pay) üzerinde çalış
        pad = 4
        box = tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad))
                    for s, dim in zip(sl, components.shape))
        region = components[box] == cid
        vcount = int(region.sum())
        if vcount < MIN_COMPONENT_VOXELS:
            continue

        com = ndimage.center_of_mass(region)
        com_full = [com[i] + box[i].start for i in range(3)]
        centroid = mask_img.TransformContinuousIndexToPhysicalPoint(
            [float(com_full[2]), float(com_full[1]), float(com_full[0])])

        z_idx = np.unique(np.nonzero(region)[0])
        diameter = max_axial_diameter(region, spacing)
        lesion = {
            "id": int(cid),
            "centroid_mm": [round(float(c), 2) for c in centroid],
            "volume_mm3": round(vcount * voxel_vol, 2),
            "diameter_mm": diameter,
            "z_extent_mm": round(float(len(z_idx) * spacing[2]), 2),
            "organ": assign_organ(region, label_arr[box], lesion_label_id),
            "voxel_count": vcount,
            "is_measurable": diameter >= thr,
            "measurability_threshold_mm": thr,
        }
        if ct_arr is not None:
            lesion["hu_mean"] = round(float(ct_arr[box][region].mean()), 1)
        # Geriye dönük uyumluluk
        lesion["is_target"] = lesion["is_measurable"]
        lesions.append(lesion)

    return lesions, components


def select_target_lesions(lesions: List[Dict[str, Any]], max_total: int = 5,
                          max_per_organ: int = 2) -> List[Dict[str, Any]]:
    """
    RECIST 1.1 hedef lezyon seçimi (bazalda bir kez yapılır):
    ölçülebilir lezyonlardan en büyükler, organ başına en fazla 2, toplam en fazla 5.
    """
    chosen, per_organ = [], {}
    for les in sorted((l for l in lesions if l["is_measurable"]),
                      key=lambda l: l["diameter_mm"], reverse=True):
        org = les.get("organ", "diger")
        if per_organ.get(org, 0) >= max_per_organ:
            continue
        chosen.append(les)
        per_organ[org] = per_organ.get(org, 0) + 1
        if len(chosen) >= max_total:
            break
    return chosen
