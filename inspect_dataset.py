"""
inspect_dataset.py
==================
HepaRECIST-AI: Hazırlanan nnU-Net v2 Eğitim Verilerini ve Organ Etiketlerini İnceleme Aracı

İşlevler:
1. Google Drive / Yerel diskteki tamamlanan hastaları listeler ve özet tablo çıkarır.
2. Her hastanın BT boyutu (Shape), Voksel aralığı (Spacing) ve 8 organ sınıfının voksel dağılımını doğrular.
3. Seçilen veya tüm hastaların en zengin aksiyel kesitinde organları renkli şeffaf maskelerle görselleştirir.
4. Google Colab içinde hem görseli hücrede ekrana basar hem de PNG olarak kaydeder.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage


# ── Organ Tanımları ve Görselleştirme Renkleri (RGBA) ────────────────────────
ORGAN_LABELS = {
    1: {"name": "Karaciğer (Liver)",       "color": [0.13, 0.77, 0.36, 0.45], "hex": "#22C55E"}, # Yeşil
    2: {"name": "Dalak (Spleen)",           "color": [0.23, 0.51, 0.96, 0.50], "hex": "#3B82F6"}, # Mavi
    3: {"name": "Böbrekler (Kidneys)",      "color": [0.94, 0.45, 0.15, 0.50], "hex": "#F97316"}, # Turuncu
    4: {"name": "Pankreas (Pancreas)",      "color": [0.95, 0.80, 0.10, 0.55], "hex": "#EAB308"}, # Sarı
    5: {"name": "Safra Kesesi (Gallb.)",    "color": [0.06, 0.73, 0.71, 0.55], "hex": "#06B6D4"}, # Camgöbeği
    6: {"name": "Mide (Stomach)",           "color": [0.66, 0.33, 0.85, 0.45], "hex": "#A855F7"}, # Mor
    7: {"name": "Aorta",                    "color": [0.88, 0.11, 0.28, 0.60], "hex": "#E11D48"}, # Kırmızı
    8: {"name": "Lezyon / Tümör (Lesion)",  "color": [1.00, 0.00, 0.50, 0.75], "hex": "#FF007F"}, # Parlak Fuşya
}


def get_dataset_paths(dataset_dir: str):
    """
    Verilen dizinden imagesTr ve labelsTr klasör yollarını çözümler.
    """
    base = Path(dataset_dir)
    # Eğer doğrudan Dataset001_AbdominalTumor verildiyse
    images_tr = base / "imagesTr"
    labels_tr = base / "labelsTr"

    if not labels_tr.exists():
        # Belki direkt nnUNet_raw altı verildi
        for candidate in base.glob("Dataset*"):
            if (candidate / "labelsTr").exists():
                return candidate / "imagesTr", candidate / "labelsTr"
        # Belki bir üst dizin verildi
        nested = base / "nnUNet_raw" / "Dataset001_AbdominalTumor"
        if (nested / "labelsTr").exists():
            return nested / "imagesTr", nested / "labelsTr"

    return images_tr, labels_tr


def list_completed_cases(dataset_dir: str):
    """
    labelsTr ve imagesTr içinde hazır olan hastaları tarar ve durumlarını listeler.
    """
    images_tr, labels_tr = get_dataset_paths(dataset_dir)

    if not labels_tr.exists():
        print(f"[HATA] labelsTr dizini bulunamadı: {labels_tr}")
        return []

    label_files = sorted(labels_tr.glob("*.nii.gz"))
    cases = []

    for lbl_file in label_files:
        case_id = lbl_file.name.replace(".nii.gz", "")
        img_file = images_tr / f"{case_id}_0000.nii.gz"

        lbl_size_mb = lbl_file.stat().st_size / (1024 * 1024)
        img_exists = img_file.exists()
        img_size_mb = (img_file.stat().st_size / (1024 * 1024)) if img_exists else 0.0

        cases.append({
            "case_id": case_id,
            "label_path": str(lbl_file),
            "image_path": str(img_file) if img_exists else None,
            "label_size_mb": lbl_size_mb,
            "image_size_mb": img_size_mb,
            "is_valid": img_exists and lbl_size_mb > 0.001 and img_size_mb > 0.001
        })

    return cases


def print_summary_table(cases: list):
    """
    Tamamlanan hastaların özet tablosunu ekrana basar.
    """
    print(f"\n{'='*78}")
    print(f"  HepaRECIST-AI: Tamamlanan Hasta / Seri İnceleme Raporu")
    print(f"{'='*78}")
    print(f"  Toplam Tespit Edilen Seri : {len(cases)}")
    valid_count = sum(1 for c in cases if c["is_valid"])
    print(f"  Eksiksiz (BT + Maske)     : {valid_count}")
    print(f"{'='*78}")
    print(f"  {'NO':<4} {'SERİ / HASTA ID':<22} {'BT (MB)':<10} {'MASKE (MB)':<12} {'DURUM'}")
    print(f"  {'-'*4} {'-'*22} {'-'*10} {'-'*12} {'-'*12}")

    for idx, c in enumerate(cases, 1):
        status_icon = "✅ HAZIR" if c["is_valid"] else "⚠️ EKSİK"
        print(f"  {idx:<4} {c['case_id']:<22} {c['image_size_mb']:>7.2f} MB {c['label_size_mb']:>9.2f} MB  {status_icon}")

    print(f"{'='*78}\n")


def inspect_single_case(case_info: dict) -> dict:
    """
    Tek bir hastanın NIfTI verilerini açıp organ bazlı voksel istatistiklerini hesaplar.
    """
    case_id = case_info["case_id"]
    lbl_img = nib.load(case_info["label_path"])
    lbl_data = np.asanyarray(lbl_img.dataobj)

    img_data = None
    spacing = lbl_img.header.get_zooms()[:3]
    shape = lbl_data.shape

    if case_info["image_path"] and os.path.exists(case_info["image_path"]):
        img_obj = nib.load(case_info["image_path"])
        img_data = np.asanyarray(img_obj.dataobj)

    voxel_counts = {}
    for lid, linfo in ORGAN_LABELS.items():
        cnt = int(np.sum(lbl_data == lid))
        voxel_counts[lid] = {
            "name": linfo["name"],
            "count": cnt,
            "volume_cm3": (cnt * np.prod(spacing)) / 1000.0
        }

    return {
        "case_id": case_id,
        "shape": shape,
        "spacing": tuple(float(s) for s in spacing),
        "voxel_counts": voxel_counts,
        "lbl_data": lbl_data,
        "img_data": img_data
    }


def visualize_case_slice(case_info: dict, output_dir: str = None, show: bool = True, slice_idx: int = None) -> str:
    """
    Hastanın BT kesitini ve üzerine 8 organın şeffaf renkli maskelerini çizer.
    """
    details = inspect_single_case(case_info)
    img_data = details["img_data"]
    lbl_data = details["lbl_data"]
    case_id = details["case_id"]

    if img_data is None:
        print(f"  [UYARI] {case_id} için BT görüntüsü bulunamadı, görselleştirilemiyor.")
        return None

    # NIfTI formatında eksenler genellikle (X, Y, Z) şeklindedir. Aksiyel kesit Z eksenidir.
    num_slices = lbl_data.shape[2]

    # En uygun kesiti belirle (Varsa lezyonun en geniş olduğu kesit; yoksa karaciğerin ortası)
    if slice_idx is None or slice_idx < 0 or slice_idx >= num_slices:
        lesion_voxels_per_slice = np.sum(lbl_data == 8, axis=(0, 1))
        if np.max(lesion_voxels_per_slice) > 0:
            best_z = int(np.argmax(lesion_voxels_per_slice))
        else:
            # Karaciğer (Label 1) ve Dalak (Label 2) toplamının en yoğun olduğu kesiti bul
            organs_per_slice = np.sum(np.isin(lbl_data, [1, 2, 3, 4]), axis=(0, 1))
            best_z = int(np.argmax(organs_per_slice)) if np.max(organs_per_slice) > 0 else num_slices // 2
    else:
        best_z = slice_idx

    ct_slice = img_data[:, :, best_z].T  # Görsel oryantasyonu için transpoze
    lbl_slice = lbl_data[:, :, best_z].T

    # 1. Yumuşak Doku HU Pencereleme: [-100, 250] HU
    min_hu, max_hu = -100.0, 250.0
    ct_norm = np.clip((ct_slice - min_hu) / (max_hu - min_hu), 0.0, 1.0)

    # 2. Şekli ve Çizim Tuvalini Hazırla
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor('#0F172A')  # Koyu şık arka plan

    # Sol Panel: Ham Aksiyel BT Kesiti
    axes[0].imshow(ct_norm, cmap='gray', origin='lower')
    axes[0].set_title(f"Aksiyel BT Kesiti (Z = {best_z}/{num_slices})\nYumuşak Doku Penceresi [-100, 250 HU]",
                      color='white', fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Sağ Panel: Multi-Organ Renkli Maske Kaplaması
    axes[1].imshow(ct_norm, cmap='gray', origin='lower')

    legend_patches = []
    # Renkli maske katmanını oluştur
    overlay = np.zeros((*lbl_slice.shape, 4), dtype=np.float32)

    for lid, info in ORGAN_LABELS.items():
        mask = (lbl_slice == lid)
        if np.any(mask):
            overlay[mask] = info["color"]

            # Karaciğer ve Lezyon için zarif kontur çizgisi ekle
            if lid in [1, 8]:
                eroded = ndimage.binary_erosion(mask)
                contour = mask ^ eroded
                contour_color = [1.0, 1.0, 0.0, 1.0] if lid == 8 else [0.2, 1.0, 0.4, 0.9]
                overlay[contour] = contour_color

            vol_cm3 = details["voxel_counts"][lid]["volume_cm3"]
            lbl_text = f"{info['name']}: {vol_cm3:.1f} cm³"
            legend_patches.append(mpatches.Patch(color=info["hex"], label=lbl_text))

    axes[1].imshow(overlay, origin='lower')
    axes[1].set_title(f"Multi-Organ AI Segmentasyon Maskesi (8 Sınıf)\n{case_id}",
                      color='white', fontsize=13, fontweight='bold', pad=10)
    axes[1].axis('off')

    if legend_patches:
        axes[1].legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.0, -0.05),
                       facecolor='#1E293B', edgecolor='#475569', labelcolor='white',
                       fontsize=9, framealpha=0.85)

    plt.tight_layout()

    # Kaydetme
    save_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = str(out_p / f"{case_id}_z{best_z:03d}_preview.png")
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"  [GÖRSEL KAYDEDİLDİ] 📸 -> {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return save_path


# ── CLI & Colab Fonksiyonları ────────────────────────────────────────────────

def run_inspection(dataset_dir: str, visualize_case: str = None, save_all_previews: bool = False, output_png_dir: str = None):
    cases = list_completed_cases(dataset_dir)
    if not cases:
        print("[BİLGİ] İncelenecek tamamlanmış hasta bulunamadı.")
        return

    print_summary_table(cases)

    out_dir = output_png_dir if output_png_dir else str(Path(dataset_dir) / "on_inceleme_gorselleri")

    if save_all_previews:
        print(f"[BİLGİ] Tüm hazır hastaların görsel önizlemeleri oluşturuluyor: {out_dir}")
        for c in cases:
            if c["is_valid"]:
                visualize_case_slice(c, output_dir=out_dir, show=False)
        print("✅ Tüm görseller tamamlandı.")

    elif visualize_case:
        # Belirli bir hasta ID veya indeks
        matched = None
        for c in cases:
            if c["case_id"].lower() == visualize_case.lower():
                matched = c
                break

        if not matched:
            # Belki numara girildi (örn: 1)
            try:
                idx = int(visualize_case) - 1
                if 0 <= idx < len(cases):
                    matched = cases[idx]
            except ValueError:
                pass

        if matched:
            print(f"\n[GÖRSELLEŞTİRİLİYOR] {matched['case_id']}...")
            visualize_case_slice(matched, output_dir=out_dir, show=True)
        else:
            print(f"[UYARI] '{visualize_case}' ID'li hasta bulunamadı.")
    else:
        # Varsayılan olarak ilk hazır hastayı ekrana bas
        first_valid = next((c for c in cases if c["is_valid"]), None)
        if first_valid:
            print(f"[ÖNİZLEME] İlk tamamlanan hasta gösteriliyor: {first_valid['case_id']}")
            visualize_case_slice(first_valid, output_dir=out_dir, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HepaRECIST-AI Hazır Veri Seti İnceleme & Görselleştirme")
    parser.add_argument("--dataset_dir", default="/content/drive/MyDrive/Egitim_Verisi_Hazir/nnUNet_raw/Dataset001_AbdominalTumor",
                        help="nnUNet_raw veri seti ana dizini")
    parser.add_argument("--case_id", default=None, help="Görselleştirilecek hasta ID'si (örn: hasta_001_t0 veya 1)")
    parser.add_argument("--save_all", action="store_true", help="Tüm hastalar için PNG kesit önizlemelerini kaydeder")
    parser.add_argument("--output_dir", default=None, help="PNG kaydedilecek hedef klasör")

    args = parser.parse_args()

    run_inspection(
        dataset_dir=args.dataset_dir,
        visualize_case=args.case_id,
        save_all_previews=args.save_all,
        output_png_dir=args.output_dir
    )
