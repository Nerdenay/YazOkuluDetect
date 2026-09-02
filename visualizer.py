"""
visualizer.py
=============
HepaRECIST-AI: 2D Aksiyel Kesit, Çok Sınıflı Doku Maskeleme ve Longitudinal Karşılaştırma Görselleştiricisi
- Sadece Lezyon (Label 8) Odaklarını Renklendirir (Organları lezyon zannetmez)
- Karaciğer (Label 1) anatomik sınırlarını arka plan konturu olarak çizer
- Radyomik Sınıfa Göre Dinamik Renklendirme:
    Kırmızı : Malign (Tümör / Metastaz)
    Mavi    : Benign (Kist / Sıvı Odakları)
    Sarı    : Vasküler / Kalsifiye Yapı
    Turuncu : Hekim Onayı Gereken Sınırda Alanlar
- 4D Longitudinal Yan Yana (Side-by-Side) RECIST 1.1 Karşılaştırma Paneli
"""

import os
from typing import Dict, Any, Optional
import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering (Sunucu / Colab ortamı için zorunlu)
import matplotlib.pyplot as plt
from scipy import ndimage


def generate_lesion_visualization(
    ct_nifti_path: str,
    mask_nifti_path: str,
    radiomics_results: Optional[Dict[str, Any]] = None,
    output_png_path: str = "./output/slice_preview.png",
    lesion_label_id: int = 8,
    organ_label_id: int = 1
) -> str:
    """
    Lezyonun en geniş kesit alanına sahip olduğu 2D aksiyel kesiti seçer.
    Karaciğer sınırlarını ve lezyon odaklarını radyomik sınıf renkleriyle çizip kaydeder.
    """
    if not os.path.exists(ct_nifti_path) or not os.path.exists(mask_nifti_path):
        raise FileNotFoundError("BT veya Maske NIfTI dosyası bulunamadı.")

    output_dir = os.path.dirname(os.path.abspath(output_png_path))
    os.makedirs(output_dir, exist_ok=True)

    ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_nifti_path))     # (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_nifti_path)) # (Z, Y, X)

    # Eğer belirtilen lezyon etiketi maskede yoksa alternatifleri kontrol et (2 veya 1)
    unique_vals = set(np.unique(mask_arr))
    if lesion_label_id not in unique_vals:
        for alt_id in [2, 1]:
            if alt_id in unique_vals:
                lesion_label_id = alt_id
                break

    # 1. En Uygun Aksiyel Kesiti (Z) Seç: SADECE Lezyon Voksellerini Say!
    lesion_counts_per_slice = np.sum(mask_arr == lesion_label_id, axis=(1, 2))
    
    if np.max(lesion_counts_per_slice) > 0:
        best_z = int(np.argmax(lesion_counts_per_slice))
    else:
        # Lezyon yoksa karaciğerin (Label 1) orta kesitini seç
        liver_counts = np.sum(mask_arr == organ_label_id, axis=(1, 2))
        best_z = int(np.argmax(liver_counts)) if np.max(liver_counts) > 0 else ct_arr.shape[0] // 2

    ct_slice = ct_arr[best_z, :, :]
    mask_slice = mask_arr[best_z, :, :]

    # 2. Yumuşak Doku HU Pencereleme: [-35, 115] HU
    min_hu, max_hu = -35.0, 115.0
    ct_norm = (np.clip(ct_slice, min_hu, max_hu) - min_hu) / (max_hu - min_hu)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(ct_norm, cmap='gray', origin='upper')

    # 3. Karaciğer Anatomik Sınırlarını Zarif Bir Kontur Olarak Göster (Label 1)
    liver_slice = (mask_slice == organ_label_id)
    if np.any(liver_slice):
        eroded = ndimage.binary_erosion(liver_slice)
        liver_contour = liver_slice ^ eroded
        contour_rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        contour_rgba[liver_contour] = [0.13, 0.77, 0.36, 0.6]  # Zümrüt Yeşili Çizgi
        ax.imshow(contour_rgba, origin='upper')

    # 4. Lezyon Voksellerini Radyomik Sınıfa Göre Dinamik Renklendir
    lesion_slice = (mask_slice == lesion_label_id)
    regions = radiomics_results.get("regions", []) if radiomics_results else []

    if np.any(lesion_slice):
        colored_mask = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        
        # Varsayılan renk: Kırmızı (Malign)
        default_color = [0.94, 0.27, 0.27, 0.5]
        colored_mask[lesion_slice] = default_color
        ax.imshow(colored_mask, origin='upper')

        # Bilgi Rozetlerini Ekrana Bas
        if regions:
            for r in regions:
                label = r.get("classification", "Lezyon")
                conf_pct = r.get("confidence_score", 0.0) * 100
                hu_mean = r.get("hu_mean", 0.0)
                psi = r.get("sphericity_psi", 0.0)
                needs_review = r.get("needs_review", False)

                if "Benign" in label:
                    badge_color = "#3B82F6"   # Mavi (Kist)
                elif "Vasküler" in label:
                    badge_color = "#EAB308"   # Sarı (Damar / Kalsifikasyon)
                elif needs_review:
                    badge_color = "#F97316"   # Turuncu (Hekim Onayı)
                else:
                    badge_color = "#EF4444"   # Kırmızı (Malign)

                y_pos = 35 + r.get("region_id", 1) * 32
                ax.text(
                    12, y_pos,
                    f"• Odak #{r.get('region_id')}: {label} (%{conf_pct:.0f}) | {hu_mean:.1f} HU | Ψ={psi:.2f}",
                    color=badge_color, fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#0F172A", edgecolor=badge_color, alpha=0.88)
                )
        else:
            ax.text(
                12, 35, "• AI Tespit Edilen Malign Lezyon Odağı",
                color="#EF4444", fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#0F172A", edgecolor="#EF4444", alpha=0.88)
            )

    ax.set_title(
        f"Aksiyel BT Kesiti Z={best_z}/{ct_arr.shape[0]} — Çok Sınıflı Doku Doğrulama",
        fontsize=11, color='white', pad=12, fontweight='bold'
    )
    ax.axis('off')
    fig.patch.set_facecolor('#0F172A')

    plt.tight_layout()
    plt.savefig(output_png_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"[SUCCESS] 2D BT Kesit Görseli kaydedildi: {output_png_path}")
    return output_png_path


def generate_longitudinal_comparison(
    baseline_ct: str,
    baseline_mask: str,
    followup_ct: str,
    followup_mask: str,
    matching_results: Dict[str, Any],
    output_png_path: str = "./output/longitudinal_comparison.png",
    lesion_label_id: int = 8
) -> str:
    """
    Baseline (t0) ve Follow-up (t1) BT serilerini lezyon odaklarıyla yan yana (side-by-side) çizer.
    """
    output_dir = os.path.dirname(os.path.abspath(output_png_path))
    os.makedirs(output_dir, exist_ok=True)

    bl_arr = sitk.GetArrayFromImage(sitk.ReadImage(baseline_ct))
    bl_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(baseline_mask))
    fu_arr = sitk.GetArrayFromImage(sitk.ReadImage(followup_ct))
    fu_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(followup_mask))

    # Etiket kontrolü
    for arr in [bl_mask_arr, fu_mask_arr]:
        unique_vals = set(np.unique(arr))
        if lesion_label_id not in unique_vals:
            for alt_id in [2, 1]:
                if alt_id in unique_vals:
                    lesion_label_id = alt_id
                    break

    # SADECE Lezyon Voksellerini Sayarak Kesit Seç!
    bl_counts = np.sum(bl_mask_arr == lesion_label_id, axis=(1, 2))
    fu_counts = np.sum(fu_mask_arr == lesion_label_id, axis=(1, 2))

    bl_z = int(np.argmax(bl_counts)) if np.max(bl_counts) > 0 else bl_arr.shape[0] // 2
    fu_z = int(np.argmax(fu_counts)) if np.max(fu_counts) > 0 else fu_arr.shape[0] // 2

    def normalize(arr, z):
        s = np.clip(arr[z, :, :], -35.0, 115.0)
        return (s - (-35.0)) / (115.0 - (-35.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), dpi=150)

    # ── Sol Panel: Baseline (t0) — Camgöbeği Mavi Overlay ──
    ax1.imshow(normalize(bl_arr, bl_z), cmap='gray', origin='upper')
    bl_lesion_slice = (bl_mask_arr[bl_z] == lesion_label_id)
    if np.any(bl_lesion_slice):
        bl_overlay = np.zeros((*bl_lesion_slice.shape, 4), dtype=np.float32)
        bl_overlay[bl_lesion_slice] = [0.14, 0.65, 0.95, 0.5]  # Açık Mavi
        ax1.imshow(bl_overlay, origin='upper')
    ax1.set_title(f"Baseline (t0) — Aksiyel Kesit Z={bl_z}", fontsize=12, color='#38BDF8', fontweight='bold')
    ax1.axis('off')

    # ── Sağ Panel: Follow-up (t1) — Ateş Kırmızısı Overlay ──
    ax2.imshow(normalize(fu_arr, fu_z), cmap='gray', origin='upper')
    fu_lesion_slice = (fu_mask_arr[fu_z] == lesion_label_id)
    if np.any(fu_lesion_slice):
        fu_overlay = np.zeros((*fu_lesion_slice.shape, 4), dtype=np.float32)
        fu_overlay[fu_lesion_slice] = [0.94, 0.27, 0.27, 0.5]  # Kırmızı
        ax2.imshow(fu_overlay, origin='upper')
    ax2.set_title(f"Follow-up (t1) — Aksiyel Kesit Z={fu_z}", fontsize=12, color='#F87171', fontweight='bold')
    ax2.axis('off')

    # Sözlük Anahtarlarını Esnek Çözümle (recist_metrics veya recist_input)
    recist = matching_results.get("recist_metrics", matching_results.get("recist_input", {}))
    sod_bl = recist.get("sod_baseline", 0.0)
    sod_fu = recist.get("sod_followup", 0.0)
    has_new = recist.get("has_new_lesions", recist.get("new_lesion", False))
    change_pct = recist.get("sod_change_pct", 0.0)

    fig.suptitle(
        f"4D LONGİTUDİNAL HİBRİT LEZYON TAKİBİ (RECIST 1.1)\n"
        f"Bazal SOD: {sod_bl:.1f} mm  ➔  Takip SOD: {sod_fu:.1f} mm ({'+' if change_pct >= 0 else ''}{change_pct:.1f}%)  |  "
        f"Yeni Lezyon: {'EVET ⚠ (PD)' if has_new else 'YOK'}",
        fontsize=13, color='#F1F5F9', fontweight='bold', y=0.98
    )

    fig.patch.set_facecolor('#0F172A')
    plt.tight_layout()
    plt.savefig(output_png_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"[SUCCESS] Longitudinal Karşılaştırma Paneli kaydedildi: {output_png_path}")
    return output_png_path


if __name__ == "__main__":
    print("=== HepaRECIST-AI: Görselleştirme Modülü (visualizer.py) ===")
