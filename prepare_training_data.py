"""
prepare_training_data.py
========================
HepaRECIST-AI: 8-Sınıflı Abdominal Organ + Manuel/Otomatik Lezyon nnU-Net v2 Hazırlama

Etiket Şeması (dataset.json):
  0: background
  1: liver
  2: spleen
  3: kidneys (Sağ + Sol birleşik)
  4: pancreas
  5: gallbladder
  6: stomach
  7: aorta
  8: lesion (Manuel etiketlenmiş NIfTI maskelerinden aktarılır)
"""

import argparse
import os
import json
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
import gc
import ctypes

def force_ram_cleanup():
    """Python çöp toplayıcısını çalıştırır ve Linux çekirdeğine RAM'i zorla iade eder."""
    gc.collect()
    try:
        # glibc heap belleğini Linux kernel'a geri verir
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# ── Sistem modüllerini import et ─────────────────────────────────────────────
try:
    from preprocess import convert_dicom_to_nifti
except ImportError as e:
    print(f"[HATA] preprocess.py bulunamadı: {e}")
    sys.exit(1)

try:
    from prepare_nnunet_data import setup_nnunet_environment, generate_dataset_json
except ImportError as e:
    print(f"[HATA] prepare_nnunet_data.py bulunamadı: {e}")
    sys.exit(1)


# ── Organ & TotalSegmentator Tanımları ───────────────────────────────────────
ABDOMINAL_ROIS = ["liver", "spleen", "kidney_right", "kidney_left", "pancreas", "gallbladder", "stomach", "aorta"]

ORGAN_LABEL_MAP = {
    "liver": 1,
    "spleen": 2,
    "kidney_right": 3,
    "kidney_left": 3,  # Sağ ve sol böbrek tek sınıf
    "pancreas": 4,
    "gallbladder": 5,
    "stomach": 6,
    "aorta": 7
}


# ── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def find_dicom_series_with_timepoints(root_dir: str) -> list:
    """
    Hastaları ve altındaki T0, T1 gibi zaman noktalarını hiyerarşik olarak bulur.
    Dönüş formatı: [{'patient_num': 1, 'tp': 't0', 'case_id': 'hasta_001_t0', 'path': '...'}]
    """
    root = Path(root_dir)
    selected_series = []

    patients = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(('.', '_'))]
    print(f"[INFO] Taranacak ana hasta klasörü sayısı: {len(patients)}")

    for p_idx, patient_dir in enumerate(patients, 1):
        candidate_dirs = {}
        for dirpath, dirnames, filenames in os.walk(patient_dir):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(('.', '_'))
                and d.upper() not in ['REPORTS', 'RA64', 'RA32', 'COMMON', 'BIN']
            ]
            valid_files = [
                f for f in filenames
                if not f.startswith(('.', '_'))
                and f.upper() not in ['DICOMDIR', 'REPORTS.TXT', 'DESCRIPT.ION']
                and not f.endswith(('.txt', '.xml', '.pdf', '.jpg', '.png', '.dll', '.exe', '.bat'))
            ]
            if len(valid_files) >= 15:
                candidate_dirs[dirpath] = len(valid_files)

        if not candidate_dirs:
            print(f"  ⚠️  {patient_dir.name} -> Geçerli DICOM serisi bulunamadı.")
            continue

        study_groups = {}
        for dirpath, count in candidate_dirs.items():
            rel = os.path.relpath(dirpath, patient_dir)
            parts = rel.split(os.sep)
            
            study_key = "t0"
            for p in parts:
                p_clean = p.lower().replace(" ", "").replace("-", "").replace("_", "")
                if "t0" in p_clean:
                    study_key = "t0"; break
                elif "t1" in p_clean:
                    study_key = "t1"; break
                elif "t2" in p_clean:
                    study_key = "t2"; break
            else:
                study_key = parts[0].lower() if parts else "t0"

            if study_key not in study_groups or count > study_groups[study_key][1]:
                study_groups[study_key] = (dirpath, count)

        for s_key, (best_dir, count) in sorted(study_groups.items()):
            case_id = f"hasta_{p_idx:03d}_{s_key}"
            selected_series.append({
                "patient_idx": p_idx,
                "timepoint": s_key,
                "case_id": case_id,
                "path": best_dir,
                "folder_name": patient_dir.name,
                "slice_count": count
            })
            print(f"  • {patient_dir.name} [{s_key.upper()}] -> {count} kesit ({case_id})")

    print(f"\n[INFO] Toplam {len(selected_series)} adet zaman serisi bulundu.\n")
    return selected_series


def run_totalsegmentator(nifti_path: str, ts_output_dir: str, fast: bool = True) -> bool:
    """
    TotalSegmentator alt işlemini RAM/VRAM taşmasını önleyen sınırlı worker ortamında çalıştırır.
    """
    os.environ["TOTALSEG_DISABLE_MP"] = "1"
    os.environ["nnUNet_def_n_proc"] = "1"

    Path(ts_output_dir).mkdir(parents=True, exist_ok=True)
    last_organ_file = Path(ts_output_dir) / "aorta.nii.gz"

    if last_organ_file.exists() and last_organ_file.stat().st_size > 1000:
        print("    [TotalSegmentator] ✅ Önceden üretilmiş batın maskeleri tam, atlanıyor.")
        return True

    import torch
    use_gpu = torch.cuda.is_available()
    device_str = "gpu" if use_gpu else "cpu"
    print(f"    [TotalSegmentator] 8 Batın organı segmentasyonu başlatılıyor (Cihaz: {device_str.upper()})...")
    if not use_gpu:
        print("    ⚠️ UYARI: GPU aktif değil! İşlem CPU üzerinde hasta başına 15-20 dakika sürebilir.")

    roi_args = ["--roi_subset"] + ABDOMINAL_ROIS
    ts_bin = shutil.which("TotalSegmentator")
    cmd = [ts_bin if ts_bin else sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"] if not ts_bin else [ts_bin]
    cmd += ["-i", nifti_path, "-o", ts_output_dir] + roi_args
    cmd += ["--device", device_str]
    # Multiprocessing ForkPoolWorker BrokenPipeError'ı tamamen engellemek için kaydetme ve resample iş parçacığını 1 yap
    cmd += ["--nr_thr_saving", "1", "--nr_thr_resamp", "1"]
    
    # --fast parametresi voxel sayısını azaltarak RAM kullanımını %85 düşürür
    if fast:
        cmd.append("--fast")

    # Colab RAM'ini korumak için worker ve thread sayılarını 1 ile sınırla
    custom_env = os.environ.copy()
    custom_env["OMP_NUM_THREADS"] = "1"
    custom_env["MKL_NUM_THREADS"] = "1"
    custom_env["nnUNet_n_proc_DA"] = "1"
    custom_env["nnUNet_def_n_proc"] = "1"
    custom_env["PYTHONUNBUFFERED"] = "1"

    ts_start = datetime.now()
    try:
        # Çıktı doğrudan terminale akar; 64KB pipe buffer kilitlenmesi (deadlock) engellenir ve canlı ilerleme çubuğu görünür
        res = subprocess.run(
            cmd,
            env=custom_env,
            timeout=1200
        )
        ts_elapsed = int((datetime.now() - ts_start).total_seconds())
        if last_organ_file.exists() and last_organ_file.stat().st_size > 1000:
            print(f"    [TotalSegmentator] ✅ Organ segmentasyonları tamamlandı ({ts_elapsed} sn).")
            return True
        else:
            print(f"    [TotalSegmentator] ⚠️ CLI tamamlanamadı (Return code: {res.returncode})")
            # Eğer OOM (-9) olduysa Python API'yi notebook içinde çalıştırma (Kernel çöker!)
            if res.returncode == -9:
                print("    [UYARI] ⚠️ Linux OOM Killer devreye girdi (RAM yetersizliği). Bellek temizleniyor...")
                force_ram_cleanup()
                return False
    except Exception as e:
        print(f"    [TotalSegmentator] CLI alt işlem hatası: {e}")
        force_ram_cleanup()
        return False

    return last_organ_file.exists() and last_organ_file.stat().st_size > 1000




def merge_organ_and_lesion_labels(ts_output_dir: str, ref_nifti_path: str, manual_lesion_path: str, merged_mask_path: str) -> dict:
    """
    Organ maskelerini (1-7) ve varsa lezyon maskesini (8) birleştirir.
    Geometri referansı olarak doğrudan orijinal nifti_path kullanılır.
    """
    ts_dir = Path(ts_output_dir)
    stats = {
        "liver_voxels": 0, "spleen_voxels": 0, "kidney_voxels": 0,
        "pancreas_voxels": 0, "lesion_voxels": 0, "has_lesion": False
    }

    if not os.path.exists(ref_nifti_path):
        print(f"    [Birleştirme] ⚠️ Referans NIfTI bulunamadı: {ref_nifti_path}")
        return stats

    ref_img = nib.load(ref_nifti_path)
    merged = np.zeros(ref_img.shape, dtype=np.uint8)

    # 1. Organ Maskelerini Birleştir (1-7)
    for roi, label_idx in ORGAN_LABEL_MAP.items():
        roi_file = ts_dir / f"{roi}.nii.gz"
        if roi_file.exists():
            roi_data = nib.load(str(roi_file)).get_fdata()
            if roi_data.shape == merged.shape:
                merged[roi_data > 0] = label_idx
                vox = int(np.sum(roi_data > 0))
                if roi == "liver": stats["liver_voxels"] = vox
                elif roi == "spleen": stats["spleen_voxels"] = vox
                elif "kidney" in roi: stats["kidney_voxels"] += vox
                elif roi == "pancreas": stats["pancreas_voxels"] = vox

    # 2. Varsa Manuel Lezyon Maskesini Birleştir (Label 8)
    if manual_lesion_path and os.path.exists(manual_lesion_path):
        lesion_img = nib.load(manual_lesion_path)
        lesion_data = lesion_img.get_fdata()
        if lesion_data.shape == merged.shape:
            merged[lesion_data > 0] = 8
            stats["lesion_voxels"] = int(np.sum(lesion_data > 0))
            stats["has_lesion"] = stats["lesion_voxels"] > 0
            print(f"    [Lezyon Entegre] ✅ Label 8 eklendi ({stats['lesion_voxels']:,} voxel)")
        else:
            print(f"    [UYARI] ⚠️ Lezyon maskesi boyutu uyuşmuyor: {lesion_data.shape} vs {merged.shape}")

    # Temiz uint8 başlık ve orijinal affine ile kaydet (scl_inter=-1024 kaymasını önler)
    out_nii = nib.Nifti1Image(merged, ref_img.affine)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, merged_mask_path)
    return stats


# ── Ana İşlem Orkestrasyonu ──────────────────────────────────────────────────

def process_all_patients(dicom_root: str, output_dir: str, manual_lesions_dir: str = None,
                         fast_mode: bool = True, dataset_id: int = 1, dataset_name: str = "AbdominalTumor"):
    start_time = datetime.now()
    output_path = Path(output_dir)

    # Google Drive I/O kilidini engellemek için yerel hızlı SSD kullanımı
    local_temp = Path("/content/temp_work") if os.path.exists("/content") else output_path / "temp_work"
    nifti_dir = local_temp / "01_nifti"
    ts_masks_dir = local_temp / "02_ts_masks"
    merged_dir = local_temp / "03_merged_labels"
    for d in [nifti_dir, ts_masks_dir, merged_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # nnU-Net v2 çıktı yolları (Google Drive veya kalıcı disk)
    nnunet_paths = setup_nnunet_environment(str(output_path), dataset_id, dataset_name)
    images_tr = nnunet_paths["imagesTr"]
    labels_tr = nnunet_paths["labelsTr"]

    import torch
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "YOK (DİKKAT: CPU MODU!)"

    print(f"\n{'='*65}")
    print(f"  HepaRECIST-AI: Multi-Organ & Tümör Veri Hazırlama Pipeline")
    print(f"{'='*65}")
    print(f"  Hesaplama Donanımı : {'CUDA GPU (' + gpu_name + ')' if has_cuda else '⚠️ CPU (DİKKAT: Aşırı yavaş!)'}")
    print(f"  Kaynak DICOM       : {dicom_root}")
    print(f"  Manuel Lezyonlar   : {manual_lesions_dir if manual_lesions_dir else 'Belirtilmedi (Sadece organlar)'}")
    print(f"  Nihai Hedef        : {nnunet_paths['dataset_dir']}")
    if not has_cuda:
        print("  ⚠️ UYARI: Colab GPU aktif değil! TotalSegmentator CPU modunda")
        print("           hasta başına 15-20 dakika sürer! Lütfen menüden T4 GPU seçin.")
    print(f"{'='*65}\n")

    series_list = find_dicom_series_with_timepoints(dicom_root)
    results = []

    for idx, item in enumerate(series_list, 1):
        case_id = item["case_id"]
        print(f"[{idx}/{len(series_list)}] {item['folder_name']} [{item['timepoint'].upper()}] → {case_id}")

        final_img = Path(images_tr) / f"{case_id}_0000.nii.gz"
        final_lbl = Path(labels_tr) / f"{case_id}.nii.gz"

        if final_img.exists() and final_lbl.exists() and final_img.stat().st_size > 1000 and final_lbl.stat().st_size > 1000:
            print(f"    [RESUME] ✅ {case_id} tamamlanmış, atlanıyor.\n")
            results.append({"id": case_id, "status": "TAMAMLANDI", "source": item["path"]})
            continue

        case_start = datetime.now()

        # 1. DICOM → NIfTI
        nifti_out = str(nifti_dir / f"{case_id}.nii.gz")
        try:
            # Eğer Drive'da imagesTr içinde NIfTI görüntüsü zaten varsa (önceki yarıda kalmış denemeden),
            # 600 DICOM dosyasını ağdan okumak yerine direkt Drive'daki NIfTI'ı kopyala (2-3 dk kazandırır)
            if final_img.exists() and final_img.stat().st_size > 1000:
                print(f"    [1/4] ⚡ NIfTI görüntüsü Drive imagesTr'de mevcut, doğrudan alınıyor...")
                if not (os.path.exists(nifti_out) and os.path.getsize(nifti_out) > 1000):
                    shutil.copy2(str(final_img), nifti_out)
                print(f"    [1/4] ✅ NIfTI hazır (Drive önbelleğinden)")
            elif not (os.path.exists(nifti_out) and os.path.getsize(nifti_out) > 1000):
                t_dcm = datetime.now()
                print(f"    [1/4] ⏳ DICOM -> NIfTI dönüştürülüyor ({item['slice_count']} kesit, Drive üzerinden okunuyor)...")
                convert_dicom_to_nifti(item["path"], str(nifti_dir), f"{case_id}.nii.gz")
                dcm_dur = int((datetime.now() - t_dcm).total_seconds())
                print(f"    [1/4] ✅ NIfTI dönüşümü OK ({dcm_dur} sn)")
            else:
                print(f"    [1/4] ✅ NIfTI dönüşümü OK (yerel SSD önbellek)")
        except Exception as e:
            print(f"    [1/4] ❌ NIfTI dönüşüm hatası: {e}\n")
            results.append({"id": case_id, "status": "HATA", "adim": "NIfTI"})
            continue

        force_ram_cleanup()

        # 2. TotalSegmentator
        ts_case_dir = str(ts_masks_dir / case_id)
        ts_ok = run_totalsegmentator(nifti_out, ts_case_dir, fast=fast_mode)
        if not ts_ok:
            print(f"    [2/4] ❌ TotalSegmentator organ maskeleri üretilemedi.\n")
            results.append({"id": case_id, "status": "HATA", "adim": "TotalSegmentator"})
            continue
        print(f"    [2/4] ✅ TotalSegmentator organ maskeleri hazır")

        # 3. Manuel Lezyon Maskesini Bul ve Birleştir
        manual_lesion_file = None
        if manual_lesions_dir:
            possible_names = [f"{case_id}.nii.gz", f"{case_id}_lesion.nii.gz", f"{item['folder_name']}_{item['timepoint']}.nii.gz"]
            for name in possible_names:
                candidate = Path(manual_lesions_dir) / name
                if candidate.exists():
                    manual_lesion_file = str(candidate)
                    break

        merged_label_path = str(merged_dir / f"{case_id}.nii.gz")
        t_merge = datetime.now()
        stats = merge_organ_and_lesion_labels(ts_case_dir, nifti_out, manual_lesion_file, merged_label_path)
        merge_dur = int((datetime.now() - t_merge).total_seconds())
        print(f"    [3/4] ✅ Etiketler birleştirildi ({merge_dur} sn) (Karaciğer: {stats['liver_voxels']:,} | Lezyon: {stats['lesion_voxels']:,} vx)")

        # 4. nnU-Net v2 Dizinine Kopyala
        try:
            t_copy = datetime.now()
            shutil.copy2(nifti_out, str(final_img))
            shutil.copy2(merged_label_path, str(final_lbl))
            copy_dur = int((datetime.now() - t_copy).total_seconds())
            case_dur = int((datetime.now() - case_start).total_seconds())
            print(f"    [4/4] ✅ nnU-Net klasörüne aktarıldı ({copy_dur} sn - Vaka süresi: {case_dur} sn).\n")
            results.append({"id": case_id, "status": "OK", "has_lesion": stats["has_lesion"]})
        except Exception as copy_err:
            print(f"    [4/4] ❌ Dosya aktarım hatası: {copy_err}\n")
            results.append({"id": case_id, "status": "HATA", "adim": "Kopyalama"})

        # 5. DISK TEMİZLİĞİ: Yerel SSD'de biriken 8 organ maskesini ve ham NIfTI'yi sil (Colab disk dolmasını önler)
        if os.path.exists(ts_case_dir):
            shutil.rmtree(ts_case_dir, ignore_errors=True)
        if os.path.exists(merged_label_path):
            os.remove(merged_label_path)
        if os.path.exists(nifti_out):
            os.remove(nifti_out)

        # 6. Bellek Temizliği (Linux glibc malloc_trim + PyTorch CUDA)
        if 'stats' in locals():
            del stats
        force_ram_cleanup()


    # dataset.json Üretimi (HepaRECIST-AI 8 Sınıflı Standart Şema)
    ok_count = sum(1 for r in results if r["status"] in ["OK", "TAMAMLANDI"])
    labels_dict = {
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

    generate_dataset_json(
        dataset_dir=nnunet_paths["dataset_dir"],
        num_training_cases=ok_count,
        labels=labels_dict
    )

    elapsed = int((datetime.now() - start_time).total_seconds())
    print(f"\n{'='*65}\n  ÖZET RAPOR\n{'='*65}")
    print(f"  Toplam Seri       : {len(series_list)}")
    print(f"  Başarılı          : {ok_count}")
    print(f"  Lezyonlu Seri     : {sum(1 for r in results if r.get('has_lesion'))}")
    print(f"  Geçen Süre        : {elapsed // 60}d {elapsed % 60}s")
    print(f"  nnU-Net Raw Dizin : {nnunet_paths['dataset_dir']}\n{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HepaRECIST-AI Veri Seti Hazırlama")
    parser.add_argument("--dicom_root", required=True, help="DICOM serilerinin ana dizini")
    parser.add_argument("--output_dir", required=True, help="Çıktı dizini (Drive dizini verilebilir)")
    parser.add_argument("--manual_lesions_dir", default=None, help="Elle çizilmiş lezyon NIfTI maskeleri dizini")
    parser.add_argument("--dataset_id", type=int, default=1, help="nnU-Net Dataset ID (örn: 1)")
    parser.add_argument("--dataset_name", default="AbdominalTumor", help="nnU-Net Dataset Adı")
    parser.add_argument("--full_quality", action="store_true", help="TotalSegmentator tam kalite modu")

    args = parser.parse_args()

    process_all_patients(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        manual_lesions_dir=args.manual_lesions_dir,
        fast_mode=not args.full_quality,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
    )
