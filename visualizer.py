import os
import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI-less (headless) rendering - sunucu ortamında gerekli
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def generate_lesion_visualization(
    ct_nifti_path: str,
    mask_nifti_path: str,
    radiomics_results: Optional[Dict[str, Any]] = None,
    output_png_path: str = "./output/slice_preview.png"
) -> str:
    """
    3D BT görüntüsü ve lezyon maskesinden, lezyonun en belirgin olduğu 2D aksiyel
    kesiti (Axial Slice) seçer. Üzerine renkli maske overlay, Radiomics sınıf etiketleri
    ve güven skorlarını çizerek PNG olarak kaydeder.

    Renk Kodlaması:
        Kırmızı  -> Malign (Tümör / Metastaz)
        Mavi     -> Benign (Kist / Sıvı Kitlesi)
        Sarı     -> Vasküler yapı / Kalsifikasyon
        Turuncu  -> Sınırda / Radyolog onayı gereken alan

    Args:
        ct_nifti_path      : Ham HU değerlerini içeren BT NIfTI dosyasının yolu.
        mask_nifti_path    : AI segmentasyon maskesi NIfTI dosyasının yolu.
        radiomics_results  : analyze_ct_and_mask_radiomics() çıktısı (opsiyonel).
                             Verilirse her lezyon adayı etiketlenir.
        output_png_path    : Kaydedilecek PNG dosyasının tam yolu.

    Returns:
        Kaydedilen PNG dosyasının tam yolu.
    """
    if not os.path.exists(ct_nifti_path):
        raise FileNotFoundError(f"BT NIfTI dosyası bulunamadı: {ct_nifti_path}")
    if not os.path.exists(mask_nifti_path):
        raise FileNotFoundError(f"Maske NIfTI dosyası bulunamadı: {mask_nifti_path}")

    output_dir = os.path.dirname(os.path.abspath(output_png_path))
    os.makedirs(output_dir, exist_ok=True)

    ct_img = sitk.ReadImage(ct_nifti_path)
    mask_img = sitk.ReadImage(mask_nifti_path)

    ct_arr = sitk.GetArrayFromImage(ct_img)    # Eksen: (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask_img) # Eksen: (Z, Y, X)

    # Lezyon piksel sayısı en fazla olan aksiyel kesiti (Z) seç
    slice_counts = np.sum(mask_arr > 0, axis=(1, 2))
    best_z = int(np.argmax(slice_counts)) if np.max(slice_counts) > 0 else ct_arr.shape[0] // 2

    ct_slice = ct_arr[best_z, :, :]
    mask_slice = mask_arr[best_z, :, :]

    # Karaciğer/yumuşak doku için HU penceresi: [-35, 115] HU
    min_hu, max_hu = -35, 115
    ct_norm = (np.clip(ct_slice, min_hu, max_hu) - min_hu) / (max_hu - min_hu)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.imshow(ct_norm, cmap='gray', origin='upper')

    regions = radiomics_results.get("regions", []) if radiomics_results else []

    if np.max(mask_slice) > 0:
        # Lezyon bölgelerine kırmızı saydam overlay uygula
        colored_mask = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        colored_mask[mask_slice > 0] = [1.0, 0.2, 0.2, 0.4]
        ax.imshow(colored_mask, origin='upper')

        if regions:
            for r in regions:
                label = r.get("classification", "Lezyon")
                conf_pct = r.get("confidence_score", 0.0) * 100
                hu_mean = r.get("hu_mean", 0.0)
                needs_review = r.get("needs_review", False)

                if "Benign" in label:
                    color = "#3B82F6"    # Mavi: Kist
                elif "Vasküler" in label:
                    color = "#EAB308"   # Sarı: Vasküler/Kireçlenme
                elif needs_review:
                    color = "#F97316"   # Turuncu: Sınırda/Şüpheli
                else:
                    color = "#EF4444"   # Kırmızı: Malign

                y_pos = 30 + r.get("region_id", 1) * 28
                ax.text(
                    12, y_pos,
                    f"• Lezyon #{r.get('region_id')}: {label}  %{conf_pct:.0f}  |  {hu_mean:.1f} HU",
                    color=color, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F172A", edgecolor=color, alpha=0.85)
                )
        else:
            ax.text(
                12, 30, "• AI Tespit Edilen Lezyon Odağı",
                color="#EF4444", fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F172A", edgecolor="#EF4444", alpha=0.85)
            )

    ax.set_title(
        f"BT Aksiyel Kesit Z={best_z}/{ct_arr.shape[0]}  —  AI Lezyon Etiketleme",
        fontsize=11, color='white', pad=10
    )
    ax.axis('off')
    fig.patch.set_facecolor('#0F172A')

    plt.tight_layout()
    plt.savefig(output_png_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"[SUCCESS] 2D BT Kesit ve Lezyon Etiket Görseli kaydedildi: {output_png_path}")
    return output_png_path


def generate_longitudinal_comparison(
    baseline_ct: str,
    baseline_mask: str,
    followup_ct: str,
    followup_mask: str,
    matching_results: Dict[str, Any],
    output_png_path: str = "./output/longitudinal_comparison.png"
) -> str:
    """
    Baseline (t0) ve Follow-up (t1) BT çiftini yan yana (side-by-side) görselleştirir.
    Her iki tarafta lezyon maskeleri renkli overlay olarak gösterilir.
    Alt başlıkta SOD değerleri ve yeni lezyon durumu yer alır.

    Args:
        baseline_ct         : Baseline BT NIfTI yolu.
        baseline_mask       : Baseline maske NIfTI yolu.
        followup_ct         : Follow-up BT NIfTI yolu.
        followup_mask       : Follow-up maske NIfTI yolu.
        matching_results    : run_longitudinal_analysis() çıktısı.
        output_png_path     : Kaydedilecek PNG dosyasının tam yolu.

    Returns:
        Kaydedilen PNG dosyasının tam yolu.
    """
    output_dir = os.path.dirname(os.path.abspath(output_png_path))
    os.makedirs(output_dir, exist_ok=True)

    bl_arr = sitk.GetArrayFromImage(sitk.ReadImage(baseline_ct))
    bl_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(baseline_mask))
    fu_arr = sitk.GetArrayFromImage(sitk.ReadImage(followup_ct))
    fu_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(followup_mask))

    bl_counts = np.sum(bl_mask_arr > 0, axis=(1, 2))
    fu_counts = np.sum(fu_mask_arr > 0, axis=(1, 2))
    bl_z = int(np.argmax(bl_counts)) if np.max(bl_counts) > 0 else bl_arr.shape[0] // 2
    fu_z = int(np.argmax(fu_counts)) if np.max(fu_counts) > 0 else fu_arr.shape[0] // 2

    def normalize(arr, z):
        s = np.clip(arr[z, :, :], -35, 115)
        return (s - (-35)) / (115 - (-35))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

    # Sol: Baseline (t0) - mavi overlay
    ax1.imshow(normalize(bl_arr, bl_z), cmap='gray', origin='upper')
    bl_overlay = np.zeros((*bl_mask_arr[bl_z].shape, 4), dtype=np.float32)
    bl_overlay[bl_mask_arr[bl_z] > 0] = [0.2, 0.6, 1.0, 0.4]
    ax1.imshow(bl_overlay, origin='upper')
    ax1.set_title(f"Baseline t0 — Kesit {bl_z}", fontsize=12, color='white')
    ax1.axis('off')

    # Sağ: Follow-up (t1) - kırmızı overlay
    ax2.imshow(normalize(fu_arr, fu_z), cmap='gray', origin='upper')
    fu_overlay = np.zeros((*fu_mask_arr[fu_z].shape, 4), dtype=np.float32)
    fu_overlay[fu_mask_arr[fu_z] > 0] = [1.0, 0.2, 0.2, 0.4]
    ax2.imshow(fu_overlay, origin='upper')
    ax2.set_title(f"Follow-up t1 — Kesit {fu_z}", fontsize=12, color='white')
    ax2.axis('off')

    recist = matching_results.get("recist_input", {})
    sod_bl = recist.get("sod_baseline", 0.0)
    sod_fu = recist.get("sod_followup", 0.0)
    has_new = recist.get("new_lesion", False)

    fig.suptitle(
        f"LONGİTUDİNAL HİBRİT LEZYON TAKİBİ\n"
        f"Baseline SOD: {sod_bl:.1f} mm  |  Follow-up SOD: {sod_fu:.1f} mm  |  Yeni Lezyon: {'EVET ⚠' if has_new else 'HAYIR'}",
        fontsize=13, color='#38BDF8', fontweight='bold', y=0.98
    )

    fig.patch.set_facecolor('#0F172A')
    plt.tight_layout()
    plt.savefig(output_png_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"[SUCCESS] Longitudinal Karşılaştırma Görseli kaydedildi: {output_png_path}")
    return output_png_path
