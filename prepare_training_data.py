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
import scipy.ndimage as ndimage
import SimpleITK as sitk
import gc
import ctypes


def force_ram_cleanup():
    """Python çöp toplayıcısını çalıştırır ve Linux çekirdeğine RAM'i zorla iade eder."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_drive_connected(target_path: str = "/content/drive/MyDrive") -> bool:
    """
    Google Drive FUSE bağlantısının canlı olup olmadığını kontrol eder.
    Hedef yol '/content/drive' içermiyorsa (örneğin yerel disk veya Windows), kontrolü atlar (True döner).
    """
    if not target_path or "/content/drive" not in str(target_path).replace("\\", "/"):
        return True
    try:
        os.stat(target_path)
        return True
    except (OSError, Exception):
        return False


# ── Sistem modüllerini import et ─────────────────────────────────────────────
try:
    from preprocess import convert_dicom_to_nifti
except ImportError as e:
    print(f"[HATA] preprocess.py bulunamadı: {e}")
    sys.exit(1)

try:
    from prepare_nnunet_data import (setup_nnunet_environment, generate_dataset_json,
                                     make_patient_level_splits)
except ImportError as e:
    print(f"[HATA] prepare_nnunet_data.py bulunamadı: {e}")
    sys.exit(1)


# ── Organ & TotalSegmentator Tanımları ───────────────────────────────────────
ABDOMINAL_ROIS = ["liver", "spleen", "kidney_right", "kidney_left", "pancreas", "gallbladder", "stomach", "aorta"]

ORGAN_LABEL_MAP = {
    "liver": 1,
    "spleen": 2,
    "kidney_right": 3,
    "kidney_left": 3,
    "pancreas": 4,
    "gallbladder": 5,
    "stomach": 6,
    "aorta": 7
}


# ── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def _score_candidate_series(dirpath: str, count: int, preferred_region: str = "abdomen") -> float:
    """
    Doğru DICOM serisini seçmek için öncelik puanı üretir.
    Hekim uyarısı: Kemik penceresi (yüksek kesit sayılı) yanlışlıkla seçilmemeli,
    batın organları için 2.5 mm Abdomen (ABD) serisi seçilmelidir.
    """
    path_lower = dirpath.lower().replace("\\", "/")
    score = float(count)

    if preferred_region == "abdomen":
        # Abdomen / Batın pozitif anahtar kelimeleri
        if any(k in path_lower for k in ["abd", "abdom", "batin", "batn", "karin", "liver"]):
            score += 20000.0
        # Kemik / Toraks / Akciğer cezaları (kemik pencere yüksek kesit sayısıyla batını ezmesin)
        if any(k in path_lower for k in ["kemik", "bone", "lung", "akciger", "parankim"]):
            score -= 15000.0
        if any(k in path_lower for k in ["scout", "surview", "topogram"]):
            score -= 50000.0
    elif preferred_region == "thorax":
        if any(k in path_lower for k in ["toraks", "thorax", "lung", "akciger", "parankim"]):
            score += 20000.0
        if any(k in path_lower for k in ["kemik", "bone"]):
            score -= 15000.0
        if any(k in path_lower for k in ["scout", "surview"]):
            score -= 50000.0

    return score


def find_dicom_series_with_timepoints(root_dir: str, preferred_region: str = "abdomen") -> list:
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

            score = _score_candidate_series(dirpath, count, preferred_region=preferred_region)
            if study_key not in study_groups or score > study_groups[study_key]["score"]:
                study_groups[study_key] = {"dirpath": dirpath, "count": count, "score": score}

        for s_key, best_info in sorted(study_groups.items()):
            best_dir = best_info["dirpath"]
            count = best_info["count"]
            case_id = f"hasta_{p_idx:03d}_{s_key}"
            selected_series.append({
                "patient_idx": p_idx,
                "timepoint": s_key,
                "case_id": case_id,
                "path": best_dir,
                "folder_name": patient_dir.name,
                "slice_count": count
            })
            print(f"  • {patient_dir.name} [{s_key.upper()}] -> {count} kesit ({case_id}) [Seçilen: {os.path.basename(best_dir)}]")

    print(f"\n[INFO] Toplam {len(selected_series)} adet zaman serisi seçildi (Bölge: {preferred_region}).\n")
    return selected_series


def run_totalsegmentator(nifti_path: str, ts_output_dir: str, fast: bool = True) -> bool:
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

    roi_args = ["--roi_subset"] + ABDOMINAL_ROIS
    ts_bin = shutil.which("TotalSegmentator")
    cmd = [ts_bin if ts_bin else sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"] if not ts_bin else [ts_bin]
    cmd += ["-i", nifti_path, "-o", ts_output_dir] + roi_args
    cmd += ["--device", device_str]
    cmd += ["--nr_thr_saving", "1", "--nr_thr_resamp", "1"]
    
    if fast:
        cmd.append("--fast")

    custom_env = os.environ.copy()
    custom_env["OMP_NUM_THREADS"] = "1"
    custom_env["MKL_NUM_THREADS"] = "1"
    custom_env["nnUNet_n_proc_DA"] = "1"
    custom_env["nnUNet_def_n_proc"] = "1"
    custom_env["PYTHONUNBUFFERED"] = "1"

    ts_start = datetime.now()
    try:
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
            if res.returncode == -9:
                print("    [UYARI] ⚠️ Linux OOM Killer devreye girdi. Bellek temizleniyor...")
            force_ram_cleanup()
            return False
    except Exception as e:
        print(f"    [TotalSegmentator] CLI alt işlem hatası: {e}")
        force_ram_cleanup()
        return False


def _resample_label_to_reference(label_path: str, ref_img: "sitk.Image") -> np.ndarray:
    """
    Etiket dosyasını referans görüntünün FİZİKSEL geometrisine (origin, spacing,
    direction) en yakın komşu ile yeniden örnekler. Yönelim farkları (RAS/LPS,
    ters kesit sırası) ve farklı kesit sayıları doğru şekilde ele alınır.
    Dönüş: (Z, Y, X) uint8 dizi.
    """
    lab = sitk.ReadImage(label_path)
    if lab.GetNumberOfComponentsPerPixel() > 1:
        lab = sitk.VectorIndexSelectionCast(lab, 0)
    lab = sitk.Cast(lab > 0, sitk.sitkUInt8) if lab.GetPixelID() not in (sitk.sitkUInt8,) else lab
    out = sitk.Resample(lab, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


def _physical_volume_ml(label_path: str) -> float:
    img = sitk.ReadImage(label_path)
    arr = sitk.GetArrayViewFromImage(img)
    sp = img.GetSpacing()
    return float((arr > 0).sum()) * sp[0] * sp[1] * sp[2] / 1000.0


def merge_organ_and_lesion_labels(ts_output_dir: str, ref_nifti_path: str, manual_lesion_path: str, merged_mask_path: str) -> dict:
    ts_dir = Path(ts_output_dir)
    stats = {
        "liver_voxels": 0, "spleen_voxels": 0, "kidney_voxels": 0,
        "pancreas_voxels": 0, "lesion_voxels": 0, "has_lesion": False,
        "has_manual_lesion_mask": False, "geometry_warning": None
    }

    if not os.path.exists(ref_nifti_path):
        print(f"    [Birleştirme] ⚠️ Referans NIfTI bulunamadı: {ref_nifti_path}")
        return stats

    ref_img = sitk.ReadImage(ref_nifti_path)
    merged = np.zeros(sitk.GetArrayViewFromImage(ref_img).shape, dtype=np.uint8)

    # 1. Organ maskeleri (1-7): fiziksel geometriye göre yeniden örnekleme
    for roi, label_idx in ORGAN_LABEL_MAP.items():
        roi_file = ts_dir / f"{roi}.nii.gz"
        if roi_file.exists():
            roi_data = _resample_label_to_reference(str(roi_file), ref_img)
            merged[roi_data > 0] = label_idx
            vox = int(np.sum(roi_data > 0))
            if roi == "liver": stats["liver_voxels"] = vox
            elif roi == "spleen": stats["spleen_voxels"] = vox
            elif "kidney" in roi: stats["kidney_voxels"] += vox
            elif roi == "pancreas": stats["pancreas_voxels"] = vox

    # 2. Elle çizilmiş lezyon maskesi (Label 8)
    if manual_lesion_path and os.path.exists(manual_lesion_path):
        stats["has_manual_lesion_mask"] = True
        lesion_data = _resample_label_to_reference(manual_lesion_path, ref_img)
        merged[lesion_data > 0] = 8
        stats["lesion_voxels"] = int(np.sum(lesion_data > 0))
        stats["has_lesion"] = stats["lesion_voxels"] > 0

        # Geometri doğrulaması: yeniden örnekleme sonrası hacim korunmalı
        src_ml = _physical_volume_ml(manual_lesion_path)
        sp = ref_img.GetSpacing()
        dst_ml = stats["lesion_voxels"] * sp[0] * sp[1] * sp[2] / 1000.0
        if src_ml > 0 and (dst_ml == 0 or abs(dst_ml - src_ml) / src_ml > 0.3):
            stats["geometry_warning"] = (f"Lezyon hacmi hizalamada değişti ({src_ml:.2f} -> {dst_ml:.2f} mL). "
                                         "Maske bu BT serisine ait olmayabilir veya FOV dışında kalıyor.")
            print(f"    [UYARI] ⚠️ {stats['geometry_warning']}")
        print(f"    [Lezyon Entegre] ✅ Label 8 eklendi ({stats['lesion_voxels']:,} voxel)")

    out = sitk.GetImageFromArray(merged)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, merged_mask_path)
    return stats


# ── Ana İşlem Orkestrasyonu ──────────────────────────────────────────────────

def process_all_patients(dicom_root: str, output_dir: str, manual_lesions_dir: str = None,
                         fast_mode: bool = True, dataset_id: int = 1, dataset_name: str = "AbdominalTumor",
                         include_unlabeled: bool = False, test_fraction: float = 0.2, seed: int = 42,
                         preferred_region: str = "abdomen"):
    """
    include_unlabeled: False (varsayılan) ise elle lezyon maskesi OLMAYAN seriler eğitime
        alınmaz. Lezyonu olan ama çizilmemiş bir seri, modele "lezyon = arka plan"
        öğretir. Yalnızca gerçekten lezyonsuz olduğu radyologca bilinen seriler için
        True yapın (veya o seriler için boş maske dosyası verin).
    test_fraction: Hasta düzeyinde ayrılan, eğitimde hiç kullanılmayan test oranı.
    preferred_region: 'abdomen' veya 'thorax'. Batın organları için kemik/toraks serileri yerine
        2.5 mm Abdomen (ABD) serisine öncelik verir.
    """
    start_time = datetime.now()
    output_path = Path(output_dir)

    local_temp = Path("/content/temp_work") if os.path.exists("/content") else output_path / "temp_work"
    nifti_dir = local_temp / "01_nifti"
    ts_masks_dir = local_temp / "02_ts_masks"
    merged_dir = local_temp / "03_merged_labels"
    for d in [nifti_dir, ts_masks_dir, merged_dir]:
        d.mkdir(parents=True, exist_ok=True)

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
    print(f"  Hedef Anatomik Alan: {preferred_region.upper()}")
    print(f"  Kaynak DICOM       : {dicom_root}")
    print(f"  Manuel Lezyonlar   : {manual_lesions_dir if manual_lesions_dir else 'Belirtilmedi (Sadece organlar)'}")
    print(f"  Nihai Hedef        : {nnunet_paths['dataset_dir']}\n{'='*65}\n")

    series_list = find_dicom_series_with_timepoints(dicom_root, preferred_region=preferred_region)
    results = []

    for idx, item in enumerate(series_list, 1):
        case_id = item["case_id"]

        # Drive FUSE kontrolü: Hedef Drive ise ve bağlantı kopmuşsa durdur
        if not is_drive_connected(str(output_path)):
            print("\n🚨 KRİTİK HATA: Google Drive FUSE bağlantısı koptu!")
            print("   İşlem veri kaybını önlemek için güvenle durduruldu.")
            print("   Lütfen Drive'ı yeniden bağlayıp scripti tekrar çalıştırın.\n")
            break

        print(f"[{idx}/{len(series_list)}] {item['folder_name']} [{item['timepoint'].upper()}] → {case_id}")

        final_img = Path(images_tr) / f"{case_id}_0000.nii.gz"
        final_lbl = Path(labels_tr) / f"{case_id}.nii.gz"

        already_completed = False
        try:
            if final_img.is_file() and final_lbl.is_file():
                if final_img.stat().st_size > 1000 and final_lbl.stat().st_size > 1000:
                    already_completed = True
        except OSError:
            already_completed = False

        if already_completed:
            print(f"    [RESUME] ✅ {case_id} tamamlanmış, atlanıyor.\n")
            results.append({"id": case_id, "status": "TAMAMLANDI", "source": item["path"],
                            "patient_idx": item["patient_idx"]})
            continue

        case_start = datetime.now()

        # 1. DICOM → NIfTI
        nifti_out = str(nifti_dir / f"{case_id}.nii.gz")
        try:
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

        # 2. TotalSegmentator Ön Hazırlık (Akıllı Z-Throttling: 520+ Kesit Koruması)
        ts_case_dir = str(ts_masks_dir / case_id)
        input_for_ts = nifti_out
        tmp_throttled_nii = None

        try:
            full_nii = nib.load(nifti_out)
            orig_shape = full_nii.shape
            # Eğer kesit sayısı > 520 ise (Colab 12GB RAM'ini patlatacak büyüklükte)
            if orig_shape[2] > 520:
                print(f"    [RAM Koruması] 🛡️ Kesit sayısı yüksek ({orig_shape[2]}). TotalSegmentator için z seyreltiliyor...")
                throttled_data = full_nii.get_fdata()[:, :, ::2]
                new_affine = full_nii.affine.copy()
                new_affine[:3, 2] *= 2.0  # Z spacing'i 2 katına çıkar
                tmp_throttled_nii = str(nifti_dir / f"{case_id}_throttled.nii.gz")
                nib.save(nib.Nifti1Image(throttled_data.astype(np.int16), new_affine), tmp_throttled_nii)
                input_for_ts = tmp_throttled_nii
                del throttled_data
                force_ram_cleanup()
        except Exception as throttle_err:
            print(f"    [UYARI] Seyreltme uygulanamadı: {throttle_err}")
            input_for_ts = nifti_out

        # TotalSegmentator'ı Çalıştır
        ts_ok = run_totalsegmentator(input_for_ts, ts_case_dir, fast=fast_mode)

        # Geçici seyreltilmiş NIfTI'yi hemen sil
        if tmp_throttled_nii and os.path.exists(tmp_throttled_nii):
            os.remove(tmp_throttled_nii)

        if not ts_ok:
            print(f"    [2/4] ❌ TotalSegmentator organ maskeleri üretilemedi.\n")
            results.append({"id": case_id, "status": "HATA", "adim": "TotalSegmentator"})
            if os.path.exists(ts_case_dir):
                shutil.rmtree(ts_case_dir, ignore_errors=True)
            if os.path.exists(nifti_out):
                os.remove(nifti_out)
            force_ram_cleanup()
            continue
        print(f"    [2/4] ✅ TotalSegmentator organ maskeleri hazır")

        # 3. Manuel Lezyon Maskesini Bul ve Birleştir (Orijinal Kesit Boyutuna Geri Büyütme)
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
        if not stats["has_manual_lesion_mask"] and not include_unlabeled:
            print(f"    [3/4] ⏭️  {case_id}: elle lezyon maskesi yok, eğitimden çıkarıldı "
                  "(lezyonsuz olduğu biliniyorsa boş maske verin veya --include_unlabeled kullanın).\n")
            results.append({"id": case_id, "status": "ETIKETSIZ", "patient_idx": item["patient_idx"]})
            for pth in (merged_label_path, nifti_out):
                if os.path.exists(pth):
                    os.remove(pth)
            shutil.rmtree(ts_case_dir, ignore_errors=True)
            force_ram_cleanup()
            continue
        if stats.get("geometry_warning"):
            results.append({"id": case_id, "status": "GEOMETRI_UYARI", "warning": stats["geometry_warning"]})
            print(f"    [3/4] ❌ {case_id} geometri uyuşmazlığı nedeniyle eğitime alınmadı.\n")
            for pth in (merged_label_path, nifti_out):
                if os.path.exists(pth):
                    os.remove(pth)
            shutil.rmtree(ts_case_dir, ignore_errors=True)
            continue
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
            results.append({"id": case_id, "status": "OK", "has_lesion": stats["has_lesion"],
                            "patient_idx": item["patient_idx"]})
        except Exception as copy_err:
            print(f"    [4/4] ❌ Dosya aktarım hatası: {copy_err}\n")
            results.append({"id": case_id, "status": "HATA", "adim": "Kopyalama"})

        # 5. DİSK TEMİZLİĞİ: Yerel SSD'deki geçici dosyaları sil
        if os.path.exists(ts_case_dir):
            shutil.rmtree(ts_case_dir, ignore_errors=True)
        if os.path.exists(merged_label_path):
            os.remove(merged_label_path)
        if os.path.exists(nifti_out):
            os.remove(nifti_out)

        # 6. Bellek Temizliği
        if 'stats' in locals():
            del stats
        force_ram_cleanup()

    # dataset.json Üretimi
    if is_drive_connected(str(output_path)):
        ok_cases = [r["id"] for r in results if r["status"] in ["OK", "TAMAMLANDI"]]
        # Hasta düzeyinde test ayrımı + çapraz doğrulama katları (aynı hastanın t0/t1
        # serileri asla eğitim ve doğrulama/test arasında bölünmez)
        split_info = make_patient_level_splits(
            nnunet_paths, ok_cases, n_folds=5, test_fraction=test_fraction, seed=seed)
        ok_count = len(split_info["train_cases"])
        labels_dict = {
            "background": 0, "liver": 1, "spleen": 2, "kidneys": 3,
            "pancreas": 4, "gallbladder": 5, "stomach": 6, "aorta": 7,
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
        print(f"  Etiketsiz (çıkarıldı): {sum(1 for r in results if r['status'] == 'ETIKETSIZ')}")
        print(f"  Geometri uyarısı  : {sum(1 for r in results if r['status'] == 'GEOMETRI_UYARI')}")
        print(f"  Eğitim / Test     : {ok_count} seri / {len(split_info['test_cases'])} seri "
              f"({len(split_info['test_patients'])} test hastası)")
        print(f"  Geçen Süre        : {elapsed // 60}d {elapsed % 60}s")
        print(f"  nnU-Net Raw Dizin : {nnunet_paths['dataset_dir']}\n{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HepaRECIST-AI Veri Seti Hazırlama")
    parser.add_argument("--dicom_root", required=True, help="DICOM serilerinin ana dizini")
    parser.add_argument("--output_dir", required=True, help="Çıktı dizini (Drive dizini verilebilir)")
    parser.add_argument("--manual_lesions_dir", default=None, help="Elle çizilmiş lezyon NIfTI maskeleri dizini")
    parser.add_argument("--dataset_id", type=int, default=1, help="nnU-Net Dataset ID (örn: 1)")
    parser.add_argument("--dataset_name", default="AbdominalTumor", help="nnU-Net Dataset Adı")
    parser.add_argument("--full_quality", action="store_true", help="TotalSegmentator tam kalite modu")
    parser.add_argument("--include_unlabeled", action="store_true",
                        help="Elle lezyon maskesi olmayan serileri de eğitime al (yalnızca lezyonsuz olduğu biliniyorsa)")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Hasta düzeyinde test oranı")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preferred_region", default="abdomen", choices=["abdomen", "thorax"],
                        help="Öncelikli taranacak seri tipi ('abdomen': 2.5mm ABD batın serileri, 'thorax': toraks serileri)")

    args = parser.parse_args()

    process_all_patients(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        manual_lesions_dir=args.manual_lesions_dir,
        fast_mode=not args.full_quality,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        include_unlabeled=args.include_unlabeled,
        test_fraction=args.test_fraction,
        seed=args.seed,
        preferred_region=args.preferred_region,
    )