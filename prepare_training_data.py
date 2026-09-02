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


# Ana batın organları listesi (TotalSegmentator)
ABDOMINAL_ROIS = ["liver", "spleen", "kidney_right", "kidney_left", "pancreas", "gallbladder", "stomach", "aorta"]

ORGAN_LABEL_MAP = {
    "liver": 1,
    "spleen": 2,
    "kidney_right": 3,
    "kidney_left": 3,  # Sağ ve sol böbrek birleşik Label 3
    "pancreas": 4,
    "gallbladder": 5,
    "stomach": 6,
    "aorta": 7,
    "lesion": 8
}

def run_totalsegmentator(nifti_path: str, ts_output_dir: str, fast: bool = True) -> bool:
    """
    TotalSegmentator'ı izole bir alt işlem (subprocess) olarak çalıştırır.
    Tüm ana batın organlarını (karaciğer, dalak, böbrekler, pankreas, safra kesesi, mide, aorta) etiketler.
    """
    Path(ts_output_dir).mkdir(parents=True, exist_ok=True)
    liver_file = Path(ts_output_dir) / "liver.nii.gz"

    if liver_file.exists() and liver_file.stat().st_size > 1000:
        print(f"    [TotalSegmentator] ✅ Önceden üretilmiş batın maskeleri mevcut, atlanıyor.")
        return True

    print(f"    [TotalSegmentator] Batın organları ve lezyon segmentasyonu başlatılıyor...")

    # 1. İzole CLI Subprocess (En kararlı ve kesintisiz yöntem)
    try:
        ts_bin = shutil.which("TotalSegmentator")
        roi_args = ["--roi_subset"] + ABDOMINAL_ROIS
        if ts_bin:
            cmd = [ts_bin, "-i", nifti_path, "-o", ts_output_dir] + roi_args + ["--quiet"]
        else:
            cmd = [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator", "-i", nifti_path, "-o", ts_output_dir] + roi_args + ["--quiet"]

        if fast:
            cmd.append("--fast")

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=1200)
        
        if liver_file.exists() and liver_file.stat().st_size > 1000:
            print(f"    [TotalSegmentator] ✅ Batın organları segmentasyonu tamamlandı.")
            return True
        else:
            err_msg = res.stderr[-300:].strip() if res.stderr else "Bilinmeyen hata"
            print(f"    [TotalSegmentator] ⚠️  CLI uyarısı: {err_msg}, Python API deneniyor...")

    except Exception as e:
        print(f"    [TotalSegmentator] CLI alt işlem hatası: {e}, Python API deneniyor...")

    # 2. Fallback: Doğrudan Python API
    try:
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        
        img = nib.load(nifti_path)
        totalsegmentator(
            input=img,
            output=Path(ts_output_dir),
            fast=fast,
            roi_subset=ABDOMINAL_ROIS,
            quiet=True,
            verbose=False
        )
        
        if liver_file.exists() and liver_file.stat().st_size > 1000:
            print(f"    [TotalSegmentator] ✅ Batın organları segmentasyonu tamamlandı (Python API).")
            return True
        else:
            print(f"    [TotalSegmentator] ❌ Batın maskeleri üretilemedi.")
            return False

    except Exception as e:
        print(f"    [TotalSegmentator] ❌ Genel hata: {e}")
        return False



def merge_organ_labels(ts_output_dir: str, merged_mask_path: str) -> dict:
    """
    TotalSegmentator'ın tüm batın organlarını tek bir 3D maskede birleştirir:
      Label 0: Arka plan
      Label 1: Karaciğer (Liver)
      Label 2: Dalak (Spleen)
      Label 3: Böbrekler (Kidneys)
      Label 4: Pankreas (Pancreas)
      Label 5: Safra Kesesi (Gallbladder)
      Label 6: Mide (Stomach)
      Label 7: Aorta (Aorta)
      Label 8: Tümör / Lezyon (Lesions)

    Returns: İstatistik dict (voxel sayıları)
    """
    import nibabel as nib
    import numpy as np

    ts_dir = Path(ts_output_dir)
    liver_path = ts_dir / "liver.nii.gz"

    stats = {
        "liver_voxels": 0, "spleen_voxels": 0, "kidney_voxels": 0,
        "pancreas_voxels": 0, "stomach_voxels": 0, "lesion_voxels": 0,
        "has_lesion": False, "total_organ_voxels": 0
    }

    if not liver_path.exists():
        print(f"    [Birleştirme] ⚠️  liver.nii.gz bulunamadı, boş maske üretiliyor")
        return stats

    ref_img = nib.load(str(liver_path))
    ref_data = ref_img.get_fdata().astype("uint8")
    merged = np.zeros_like(ref_data, dtype="uint8")

    # Tüm batın organlarını sırayla katmana yaz
    for roi, label_idx in ORGAN_LABEL_MAP.items():
        if roi == "lesion":
            continue
        roi_file = ts_dir / f"{roi}.nii.gz"
        if roi_file.exists():
            roi_data = nib.load(str(roi_file)).get_fdata()
            merged[roi_data > 0] = label_idx
            vox_count = int(np.sum(roi_data > 0))
            if roi == "liver":
                stats["liver_voxels"] = vox_count
            elif roi == "spleen":
                stats["spleen_voxels"] = vox_count
            elif "kidney" in roi:
                stats["kidney_voxels"] += vox_count
            elif roi == "pancreas":
                stats["pancreas_voxels"] = vox_count
            elif roi == "stomach":
                stats["stomach_voxels"] = vox_count

    # Varsa lezyonları en üst katmana Label 8 olarak bas
    tumor_candidates = ["liver_tumor.nii.gz", "tumor.nii.gz", "lesion.nii.gz"]
    for tc in tumor_candidates:
        t_file = ts_dir / tc
        if t_file.exists():
            t_data = nib.load(str(t_file)).get_fdata()
            merged[t_data > 0] = 8
            stats["lesion_voxels"] += int(np.sum(t_data > 0))
            stats["has_lesion"] = True

    stats["total_organ_voxels"] = int(np.sum(merged > 0))

    print(f"    [Birleştirme] Karaciğer: {stats['liver_voxels']:,} | Dalak: {stats['spleen_voxels']:,} | "
          f"Böbrek: {stats['kidney_voxels']:,} | Pankreas: {stats['pancreas_voxels']:,} | "
          f"Lezyon: {stats['lesion_voxels']:,} voxel")

    nib.save(nib.Nifti1Image(merged, ref_img.affine, ref_img.header), merged_mask_path)
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
            stats = {"liver_voxels": 0, "lesion_voxels": 0, "has_lesion": False}
            print(f"    [3/3] ❌ Segmentasyon başarısız olduğu için etiket üretilemedi.")
        # 4. nnU-Net v2 formatına kopyala ────────────────────────────────────
        # imagesTr: hasta_001_0000.nii.gz (BT kanalı)
        # labelsTr: hasta_001.nii.gz      (maske)
        if ts_ok and stats["liver_voxels"] > 0:
            try:
                shutil.copy2(nifti_out, str(Path(images_tr) / f"{patient_id}_0000.nii.gz"))
                shutil.copy2(merged_label_path, str(Path(labels_tr) / f"{patient_id}.nii.gz"))
                print(f"    [4/4] ✅ nnU-Net klasörüne aktarıldı.")
            except Exception as copy_err:
                print(f"    [4/4] ❌ nnU-Net kopyalama hatası: {copy_err}")
                results.append({"id": patient_id, "status": "HATA", "adim": "Kopyalama"})
                print()
                continue
        else:
            print(f"    [4/4] ⚠️  Segmentasyon başarısız olduğu için nnU-Net klasörüne kopyalanmadı.")


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
        labels={
            "background": 0,
            "liver": 1,
            "spleen": 2,
            "kidneys": 3,
            "pancreas": 4,
            "gallbladder": 5,
            "stomach": 6,
            "aorta": 7,
            "lesion": 8
        }
    )

    # ── Özet rapor ────────────────────────────────────────────────────────────
    elapsed = int((datetime.now() - start_time).total_seconds())
    lesion_cases = sum(1 for r in results if r.get("has_lesion"))

    print(f"\n{'='*60}")
    print(f"  ÖZET RAPOR — Multi-Organ Abdominal Cavity Dataset")
    print(f"{'='*60}")
    print(f"  Toplam seri        : {len(series_list)}")
    print(f"  Başarılı           : {ok_count}")
    print(f"  Lezyon tespit      : {lesion_cases} hastada")
    print(f"  Hata/Uyarı         : {len(results) - ok_count}")
    print(f"  Toplam süre        : {elapsed // 60}d {elapsed % 60}s")
    print(f"\n  nnU-Net klasörü    : {nnunet_paths['dataset_dir']}")
    print(f"    imagesTr/        : {ok_count} dosya")
    print(f"    labelsTr/        : {ok_count} dosya")
    print(f"    dataset.json     : ✅ (8 Abdominal Sınıf + Lezyon)")
    print(f"\n  ⚠️  Maskeleri 3D Slicer / ITK-SNAP ortamında kontrol edebilirsiniz!")
    print(f"{'='*60}\n")

    errors = [r for r in results if r["status"] != "OK"]
    if errors:
        print("Sorunlu hastalar:")
        for e in errors:
            print(f"  {e['id']}: {e['status']}")

    # RunPod/Colab eğitim komutunu göster
    print("Eğitimi başlatmak için Colab/RunPod'da çalıştırın:")
    print(f"  nnUNetv2_plan_and_preprocess -d {dataset_id:03d} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {dataset_id:03d} 3d_fullres 0")
    print()

    return nnunet_paths["dataset_dir"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YazOkuluDetect — Otomatik Multi-Organ Abdominal Eğitim Verisi Hazırlama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Tek hasta test:
  python prepare_training_data.py --dicom_root ./test_dicom --output_dir ./test_output

  # Tüm hastalar (Multi-Organ Abdominal Cavity):
  python prepare_training_data.py --dicom_root D:/hastalar --output_dir D:/egitim_verisi
        """
    )
    parser.add_argument("--dicom_root",   required=True,       help="DICOM klasörlerinin ana dizini")
    parser.add_argument("--output_dir",   required=True,       help="Çıktı dizini")
    parser.add_argument("--dataset_id",   type=int, default=1, help="nnU-Net dataset ID (varsayılan: 1)")
    parser.add_argument("--dataset_name", default="AbdominalTumor", help="Dataset adı (varsayılan: AbdominalTumor)")
    parser.add_argument("--full_quality", action="store_true", help="TotalSegmentator tam kalite (yavaş)")

    args = parser.parse_args()

    process_all_patients(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        fast_mode=not args.full_quality,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
    )

