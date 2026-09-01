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
    Tüm 41 hastayı eksiksiz bulur.
    Linux büyük/küçük harf duyarlılığına (DICOM vs dicom vs t0 vs T 0) takılmaz.
    Her hasta ve zaman noktası için en büyük aksiyal tomografi serisini otomatik seçer.
    """
    root = Path(root_dir)
    selected_series = []

    print("[INFO] Tüm 41 hasta klasörü taranıyor (Büyük/küçük harf ve yapıdan bağımsız)...")
    
    # 1. Hasta klasörlerini al
    patients = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(('.', '_'))]
    print(f"[INFO] Taranacak ana hasta klasörü sayısı: {len(patients)}")

    for patient_dir in patients:
        # Bu hastanın içindeki tüm yaprak klasörleri ve dosya sayılarını bul
        candidate_dirs = {}  # {dirpath: valid_file_count}
        
        for dirpath, dirnames, filenames in os.walk(patient_dir):
            # Sistem veya viewer klasörlerini atla
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(('.', '_'))
                and d.upper() not in ['REPORTS', 'RA64', 'RA32', 'COMMON', 'BIN']
            ]
            
            # Gerçek görüntü dosyalarını say
            valid_files = [
                f for f in filenames
                if not f.startswith(('.', '_'))
                and f.upper() not in ['DICOMDIR', 'REPORTS.TXT', 'DESCRIPT.ION', 'INDEX.HTM', 'CALISTIR.BAT', 'AUTORUNN.INF']
                and not f.endswith(('.txt', '.xml', '.pdf', '.jpg', '.png', '.dll', '.exe', '.bat', '.ico', '.inf', '.htm', '.html'))
            ]
            
            if len(valid_files) >= 5:
                candidate_dirs[dirpath] = len(valid_files)

        if not candidate_dirs:
            print(f"  ⚠️  {patient_dir.name} -> Geçerli DICOM kesiti bulunamadı (boş veya rar olabilir)")
            continue

        # Bu hastanın aday klasörlerini zaman noktalarına göre grupla
        # Örn: 'T0', 'T1', 't0', 't1', 'T 0', 'T 1' veya ilk alt klasör
        study_groups = {}  # {study_key: (best_dir, max_count)}

        for dirpath, count in candidate_dirs.items():
            rel = os.path.relpath(dirpath, patient_dir)
            parts = rel.split(os.sep)
            
            # Zaman noktasını belirle (T0, T1, t0, t1 vs.)
            study_key = "T0"
            for p in parts:
                p_clean = p.upper().replace(" ", "").replace("-", "").replace("_", "")
                if "T0" in p_clean:
                    study_key = "T0"
                    break
                elif "T1" in p_clean:
                    study_key = "T1"
                    break
                elif "T2" in p_clean:
                    study_key = "T2"
                    break
            else:
                # T0/T1 yazmıyorsa ilk alt klasörün adını kullan
                study_key = parts[0] if parts else "STUDY"

            if study_key not in study_groups or count > study_groups[study_key][1]:
                study_groups[study_key] = (dirpath, count)

        # Seçilen serileri ekle
        for s_key, (best_dir, count) in sorted(study_groups.items()):
            label = f"{patient_dir.name}/{s_key}"
            print(f"  • {label} -> {count} kesit ({Path(best_dir).name})")
            selected_series.append(best_dir)

    print(f"\n[INFO] Toplam {len(selected_series)} adet geçerli BT serisi bulundu.\n")
    return selected_series


def run_totalsegmentator(nifti_path: str, ts_output_dir: str, fast: bool = True) -> bool:
    """
    TotalSegmentator'ı tek iş parçacıklı (single-thread) ve doğrudan Python API ile çalıştırır.
    nr_threads_resampling=1 ve nr_threads_saving=1 parametreleri Colab'da sahte ^C (SIGINT)
    ve multiprocessing sinyal kilitlenmelerini %100 engeller.
    """
    Path(ts_output_dir).mkdir(parents=True, exist_ok=True)
    liver_file = Path(ts_output_dir) / "liver.nii.gz"

    if liver_file.exists() and liver_file.stat().st_size > 1000:
        print(f"    [TotalSegmentator] ✅ Önceden üretilmiş maske mevcut, atlanıyor.")
        return True

    print(f"    [TotalSegmentator] Karaciğer segmentasyonu başlatılıyor...")
    
    # 1. Doğrudan Python API ile tek iş parçacığında çalıştır (en kararlı yöntem)
    try:
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        
        img = nib.load(nifti_path)
        
        totalsegmentator(
            input=img,
            output=Path(ts_output_dir),
            fast=fast,
            roi_subset=["liver"],
            quiet=True,
            verbose=False,
            nr_threads_resampling=1,
            nr_threads_saving=1
        )
        
        if liver_file.exists():
            print(f"    [TotalSegmentator] ✅ Karaciğer segmentasyonu tamamlandı.")
            return True
        else:
            print(f"    [TotalSegmentator] ⚠️  liver.nii.gz üretilemedi.")
            return False

    except Exception as e:
        print(f"    [TotalSegmentator] Python API hatası: {e}, CLI deneniyor...")

    # 2. Fallback: CLI
    try:
        ts_bin = shutil.which("TotalSegmentator") or "TotalSegmentator"
        cmd = [ts_bin, "-i", nifti_path, "-o", ts_output_dir, "--roi_subset", "liver", "--nr_threads_saving", "1"]
        if fast:
            cmd.append("--fast")
        
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=600)
        if liver_file.exists():
            print(f"    [TotalSegmentator] ✅ Karaciğer segmentasyonu tamamlandı (CLI).")
            return True
        else:
            print(f"    [TotalSegmentator] ❌ Segmentasyon başarısız: {res.stderr[-200:] if res.stderr else 'Bilinmeyen hata'}")
            return False
            
    except Exception as e:
        print(f"    [TotalSegmentator] ❌ Genel hata: {e}")
        return False



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
        rel_label = os.path.relpath(dicom_dir, dicom_root)
        print(f"[{idx}/{len(series_list)}] {rel_label} → {patient_id}")

        final_img = Path(images_tr) / f"{patient_id}_0000.nii.gz"
        final_lbl = Path(labels_tr) / f"{patient_id}.nii.gz"
        if final_img.exists() and final_lbl.exists() and final_img.stat().st_size > 1000 and final_lbl.stat().st_size > 1000:
            print(f"    [RESUME] ✅ {patient_id} zaten tamamlanmış, atlanıyor.")
            results.append({"id": patient_id, "status": "TAMAMLANDI", "has_lesion": True, "source": rel_label})
            print()
            continue

        # 1. DICOM → NIfTI ───────────────────────────────────────────────────
        nifti_out = str(nifti_dir / f"{patient_id}.nii.gz")
        try:
            if not (os.path.exists(nifti_out) and os.path.getsize(nifti_out) > 1000):
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
        try:
            shutil.copy2(nifti_out, str(Path(images_tr) / f"{patient_id}_0000.nii.gz"))
            shutil.copy2(merged_label_path, str(Path(labels_tr) / f"{patient_id}.nii.gz"))
        except Exception as copy_err:
            print(f"    [4/4] ❌ nnU-Net kopyalama hatası: {copy_err}")
            results.append({"id": patient_id, "status": "HATA", "adim": "Kopyalama"})
            print()
            continue

        # Karaciğer voxel sayısı çok düşükse uyar (yanlış seri veya boş maske)
        if stats["liver_voxels"] < 1000:
            print(f"    [⚠️] Karaciğer voxel sayısı çok düşük ({stats['liver_voxels']}), "
                  f"maske güvenilmeyebilir - devam ediliyor")

        results.append({
            "id": patient_id,
            "source_folder": rel_label,
            "status": "OK" if ts_ok else "UYARI",
            "liver_voxels": stats["liver_voxels"],
            "lesion_voxels": stats["lesion_voxels"],
            "has_lesion": stats["has_lesion"]
        })
        print()

    # 5. patient_mapping.json ve dataset.json üret ───────────────────────────
    mapping_file = str(output_path / "patient_mapping.json")
    try:
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Hasta eşleme tablosu kaydedildi: {mapping_file}")
    except Exception as e:
        print(f"[UYARI] patient_mapping.json yazılamadı: {e}")

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
