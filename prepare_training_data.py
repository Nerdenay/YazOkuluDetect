"""
prepare_training_data.py
========================
TEK KOMUTLA tüm eğitim verisi hazırlama scripti.
prepare_nnunet_data.py ve preprocess.py ile tam uyumlu.

Kullanım:
    # Test (1 hasta, hızlı):
    python prepare_training_data.py --dicom_root ./test_dicom --output_dir ./training_data

    # Tüm hastalar:
    python prepare_training_data.py --dicom_root D:/hastalar --output_dir D:/egitim_verisi

    # Tam kalite maske (yavaş):
    python prepare_training_data.py --dicom_root D:/hastalar --output_dir D:/egitim_verisi --full_quality

Çıktı yapısı (nnU-Net v2 standardı):
    <output_dir>/
      ├── 01_nifti/           ← Ham NIfTI dosyaları
      ├── 02_ts_masks/        ← TotalSegmentator ham çıktıları
      └── nnUNet_raw/
          └── Dataset001_LiverLesion/
              ├── imagesTr/   ← hasta_001_0000.nii.gz ...
              ├── labelsTr/   ← hasta_001.nii.gz ...
              ├── imagesTs/   ← Test seti (boş bırakılabilir)
              └── dataset.json
"""

import argparse
import os
import json
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime


# ── Mevcut sistem modüllerini import et ──────────────────────────────────────
try:
    from preprocess import convert_dicom_to_nifti
except ImportError as e:
    print(f"[HATA] preprocess.py bulunamadı: {e}")
    print("Proje kök klasöründen çalıştırdığınızdan emin olun.")
    sys.exit(1)

try:
    from prepare_nnunet_data import setup_nnunet_environment, generate_dataset_json, auto_scan_and_convert
except ImportError as e:
    print(f"[HATA] prepare_nnunet_data.py bulunamadı: {e}")
    sys.exit(1)


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def find_dicom_series(root_dir: str) -> list:
    """
    Verilen klasörün altındaki tüm DICOM serilerini bulur.
    - .dcm uzantılı dosyalar içeren klasörleri tarar
    - Sectra PACS DICOMDIR yapısını da destekler
    """
    series_dirs = []
    root = Path(root_dir)

    for path in sorted(root.rglob("*.dcm")):
        parent = str(path.parent)
        if parent not in series_dirs:
            series_dirs.append(parent)

    # DICOMDIR desteği (Sectra PACS exportu)
    for path in sorted(root.rglob("DICOMDIR")):
        parent = str(path.parent)
        if parent not in series_dirs:
            series_dirs.append(parent)

    # Eğer alt klasörlerde .dcm yoksa, doğrudan alt klasörleri dene
    if not series_dirs:
        series_dirs = [str(d) for d in root.iterdir() if d.is_dir()]

    return sorted(series_dirs)


def run_totalsegmentator(nifti_path: str, ts_output_dir: str, fast: bool = True) -> bool:
    """
    TotalSegmentator CLI entry point'ini subprocess olarak çalıştırır.
    Linux (Colab), Windows (.venv) ve tüm ortamlarda shutil.which ile otomatik bulur.
    """
    Path(ts_output_dir).mkdir(parents=True, exist_ok=True)

    # 1. TotalSegmentator CLI yolunu bul (Windows ve Linux/Colab uyumlu)
    ts_bin = shutil.which("TotalSegmentator")
    if not ts_bin:
        scripts_dir = Path(sys.executable).parent
        for candidate in ["TotalSegmentator", "TotalSegmentator.exe"]:
            if (scripts_dir / candidate).exists():
                ts_bin = str(scripts_dir / candidate)
                break

    def run_cmd(cmd, label):
        if ts_bin:
            full_cmd = [ts_bin] + cmd
        else:
            full_cmd = [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"] + cmd
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    [TotalSegmentator] ⚠️  {label} hatası: {result.stderr[-300:] if result.stderr else ''}")
            return False
        return True

    # Adım 1: Karaciğer segmentasyonu (fast modda çalışır)
    print(f"    [TotalSegmentator] Adım 1/2: Karaciğer segmentasyonu...")
    liver_cmd = ["-i", nifti_path, "-o", ts_output_dir, "--roi_subset", "liver"]
    if fast:
        liver_cmd.append("--fast")
    liver_ok = run_cmd(liver_cmd, "Karaciğer")

    if not liver_ok:
        # Python API fallback - sadece liver
        try:
            from totalsegmentator.python_api import totalsegmentator
            import nibabel as nib
            img = nib.load(nifti_path)
            totalsegmentator(img, Path(ts_output_dir), fast=fast,
                             roi_subset=["liver"], quiet=False)
            liver_ok = True
            print(f"    [TotalSegmentator] ✅ Karaciğer OK (Python API)")
        except Exception as e:
            print(f"    [TotalSegmentator] ❌ Python API hatası: {e}")
            return False

    # Adım 2: Karaciğer tümörü (opsiyonel)
    print(f"    [TotalSegmentator] Adım 2/2: Tümör tespiti (opsiyonel)...")
    tumor_cmd = ["-i", nifti_path, "-o", ts_output_dir,
                 "--roi_subset", "liver_tumor", "--task", "total"]
    tumor_ok = run_cmd(tumor_cmd, "Tümör")
    if not tumor_ok:
        print(f"    [TotalSegmentator] ℹ️  Tümör modeli atlandı → karaciğer parankim maskesi kullanılacak")

    print(f"    [TotalSegmentator] ✅ Segmentasyon tamamlandı")
    return liver_ok



def merge_organ_labels(ts_output_dir: str, merged_mask_path: str) -> dict:
    """
    TotalSegmentator'ın ayrı organ NIfTI dosyalarını birleştirir:
      Label 0: Arka plan
      Label 1: Karaciğer parankimi
      Label 2: Lezyon / Tümör

    Returns: İstatistik dict (voxel sayıları)
    """
    import nibabel as nib
    import numpy as np

    ts_dir = Path(ts_output_dir)
    liver_path = ts_dir / "liver.nii.gz"
    tumor_path = ts_dir / "liver_tumor.nii.gz"

    stats = {"liver_voxels": 0, "lesion_voxels": 0, "has_lesion": False}

    if not liver_path.exists():
        print(f"    [Birleştirme] ⚠️  liver.nii.gz bulunamadı, boş maske üretiliyor")
        return stats

    liver_img = nib.load(str(liver_path))
    liver_data = liver_img.get_fdata().astype("uint8")
    merged = np.zeros_like(liver_data, dtype="uint8")
    merged[liver_data > 0] = 1
    stats["liver_voxels"] = int(np.sum(liver_data > 0))

    if tumor_path.exists():
        tumor_data = nib.load(str(tumor_path)).get_fdata().astype("uint8")
        merged[tumor_data > 0] = 2
        stats["lesion_voxels"] = int(np.sum(tumor_data > 0))
        stats["has_lesion"] = stats["lesion_voxels"] > 0

    print(f"    [Birleştirme] Karaciğer: {stats['liver_voxels']:,} voxel | "
          f"Lezyon: {stats['lesion_voxels']:,} voxel "
          f"({'✅ Bulundu' if stats['has_lesion'] else '⚪ Temiz'})")

    nib.save(nib.Nifti1Image(merged, liver_img.affine, liver_img.header), merged_mask_path)
    return stats


# ── Ana işlem fonksiyonu ─────────────────────────────────────────────────────

def process_all_patients(dicom_root: str, output_dir: str, fast_mode: bool = True,
                         dataset_id: int = 1, dataset_name: str = "LiverLesion"):
    """
    Ana orkestrasyon fonksiyonu.
    Her DICOM serisi için:
      1. DICOM → NIfTI (preprocess.py)
      2. TotalSegmentator ile otomatik maske
      3. Organ etiketlerini birleştir
      4. nnU-Net v2 formatına aktar (prepare_nnunet_data.py)
      5. dataset.json üret
    """
    start_time = datetime.now()
    output_path = Path(output_dir)

    # Ara klasörler
    nifti_dir     = output_path / "01_nifti"
    ts_masks_dir  = output_path / "02_ts_masks"
    merged_dir    = output_path / "03_merged_labels"
    for d in [nifti_dir, ts_masks_dir, merged_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # nnU-Net v2 klasör hiyerarşisi (prepare_nnunet_data.py ile uyumlu)
    nnunet_paths = setup_nnunet_environment(str(output_path), dataset_id, dataset_name)
    images_tr = nnunet_paths["imagesTr"]
    labels_tr = nnunet_paths["labelsTr"]

    print(f"\n{'='*60}")
    print(f"  YazOkuluDetect — Otomatik Eğitim Verisi Hazırlama")
    print(f"{'='*60}")
    print(f"  Kaynak DICOM : {dicom_root}")
    print(f"  Çıktı        : {output_dir}")
    print(f"  Dataset      : {nnunet_paths['dataset_name']}")
    print(f"  Mod          : {'Hızlı (--fast)' if fast_mode else 'Tam kalite'}")
    print(f"{'='*60}\n")

    series_list = find_dicom_series(dicom_root)
    print(f"Bulunan DICOM serisi: {len(series_list)}\n")

    results = []

    for idx, dicom_dir in enumerate(series_list, 1):
        patient_id = f"hasta_{idx:03d}"
        print(f"[{idx}/{len(series_list)}] {Path(dicom_dir).name} → {patient_id}")

        # 1. DICOM → NIfTI ───────────────────────────────────────────────────
        nifti_out = str(nifti_dir / f"{patient_id}.nii.gz")
        try:
            convert_dicom_to_nifti(dicom_dir, str(nifti_dir), f"{patient_id}.nii.gz")
            if not os.path.exists(nifti_out):
                raise FileNotFoundError(f"NIfTI dosyası oluşturulamadı: {nifti_out}")
            print(f"    [1/3] ✅ NIfTI dönüşümü OK")
        except Exception as e:
            print(f"    [1/3] ❌ NIfTI dönüşümü başarısız: {e}")
            results.append({"id": patient_id, "status": "HATA", "adim": "NIfTI"})
            print()
            continue

        # 2. TotalSegmentator ─────────────────────────────────────────────────
        ts_out_dir = str(ts_masks_dir / patient_id)
        ts_ok = run_totalsegmentator(nifti_out, ts_out_dir, fast=fast_mode)

        # 3. Etiketleri birleştir ─────────────────────────────────────────────
        merged_label_path = str(merged_dir / f"{patient_id}.nii.gz")

        if ts_ok:
            stats = merge_organ_labels(ts_out_dir, merged_label_path)
            print(f"    [3/3] ✅ Etiket birleştirme OK")
        else:
            # TotalSegmentator başarısız → boş maske (karaciğer de yok)
            import nibabel as nib, numpy as np
            ref = nib.load(nifti_out)
            nib.save(nib.Nifti1Image(np.zeros(ref.shape, dtype="uint8"), ref.affine), merged_label_path)
            stats = {"liver_voxels": 0, "lesion_voxels": 0, "has_lesion": False}
            print(f"    [3/3] ⚠️  Boş maske üretildi (TotalSegmentator başarısız)")

        # 4. nnU-Net v2 formatına kopyala ────────────────────────────────────
        # imagesTr: hasta_001_0000.nii.gz (BT kanalı)
        # labelsTr: hasta_001.nii.gz      (maske)
        shutil.copy2(nifti_out, str(Path(images_tr) / f"{patient_id}_0000.nii.gz"))
        shutil.copy2(merged_label_path, str(Path(labels_tr) / f"{patient_id}.nii.gz"))

        results.append({
            "id": patient_id,
            "status": "OK" if ts_ok else "UYARI",
            "liver_voxels": stats["liver_voxels"],
            "lesion_voxels": stats["lesion_voxels"],
            "has_lesion": stats["has_lesion"]
        })
        print()

    # 5. dataset.json üret (prepare_nnunet_data.py fonksiyonu) ───────────────
    ok_count = sum(1 for r in results if r["status"] == "OK")
    generate_dataset_json(
        dataset_dir=nnunet_paths["dataset_dir"],
        num_training_cases=ok_count,
        labels={"background": 0, "liver": 1, "lesion": 2}
    )

    # ── Özet rapor ────────────────────────────────────────────────────────────
    elapsed = int((datetime.now() - start_time).total_seconds())
    lesion_cases = sum(1 for r in results if r.get("has_lesion"))

    print(f"\n{'='*60}")
    print(f"  ÖZET RAPOR")
    print(f"{'='*60}")
    print(f"  Toplam seri        : {len(series_list)}")
    print(f"  Başarılı           : {ok_count}")
    print(f"  Lezyon tespit      : {lesion_cases} hastada")
    print(f"  Hata/Uyarı         : {len(results) - ok_count}")
    print(f"  Toplam süre        : {elapsed // 60}d {elapsed % 60}s")
    print(f"\n  nnU-Net klasörü    : {nnunet_paths['dataset_dir']}")
    print(f"    imagesTr/        : {ok_count} dosya")
    print(f"    labelsTr/        : {ok_count} dosya")
    print(f"    dataset.json     : ✅")
    print(f"\n  ⚠️  Maskeleri 3D Slicer'da kontrol ettirmeyi unutmayın!")
    print(f"{'='*60}\n")

    errors = [r for r in results if r["status"] != "OK"]
    if errors:
        print("Sorunlu hastalar:")
        for e in errors:
            print(f"  {e['id']}: {e['status']}")

    # RunPod eğitim komutunu göster
    print("Eğitimi başlatmak için RunPod'da çalıştırın:")
    print(f"  nnUNetv2_plan_and_preprocess -d {dataset_id:03d} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {dataset_id:03d} 3d_fullres 0")
    print()

    return nnunet_paths["dataset_dir"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YazOkuluDetect — Otomatik Eğitim Verisi Hazırlama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Tek hasta test:
  python prepare_training_data.py --dicom_root ./test_dicom --output_dir ./test_output

  # Tüm hastalar:
  python prepare_training_data.py --dicom_root D:/hastalar --output_dir D:/egitim_verisi

  # Tam kalite maske (yavaş):
  python prepare_training_data.py --dicom_root D:/hastalar --output_dir D:/egitim_verisi --full_quality
        """
    )
    parser.add_argument("--dicom_root",   required=True,       help="DICOM klasörlerinin ana dizini")
    parser.add_argument("--output_dir",   required=True,       help="Çıktı dizini")
    parser.add_argument("--dataset_id",   type=int, default=1, help="nnU-Net dataset ID (varsayılan: 1)")
    parser.add_argument("--dataset_name", default="LiverLesion", help="Dataset adı (varsayılan: LiverLesion)")
    parser.add_argument("--full_quality", action="store_true", help="TotalSegmentator tam kalite (yavaş)")

    args = parser.parse_args()

    process_all_patients(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        fast_mode=not args.full_quality,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
    )
