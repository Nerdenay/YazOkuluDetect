"""
prepare_nnunet_data.py
======================
HepaRECIST-AI: nnU-Net v2 Ortam Kurulumu, Dataset Yönetimi ve Bulut Eğitim Scripti Üretici
- nnUNet_raw, nnUNet_preprocessed, nnUNet_results dizin hiyerarşisi
- Multi-Organ Abdominal Şema (8 Organ + 1 Lezyon) Destekli dataset.json üretimi
- RunPod / Vast.ai GPU sunucuları için 5-Fold Cross Validation Otomasyon Scripti (.sh)
"""

import os
import sys
import shutil
import json
from typing import List, Dict, Optional, Any


def setup_nnunet_environment(base_dir: str, dataset_id: int = 1, dataset_name: str = "AbdominalTumor") -> Dict[str, str]:
    """
    nnU-Net v2 için gerekli tüm ortam dizinlerini ve DatasetXXX_Name klasör hiyerarşisini kurar.
    """
    formatted_dataset_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    
    raw_dir = os.path.join(base_dir, "nnUNet_raw", formatted_dataset_name)
    images_tr = os.path.join(raw_dir, "imagesTr")
    labels_tr = os.path.join(raw_dir, "labelsTr")
    images_ts = os.path.join(raw_dir, "imagesTs")
    
    preprocessed_dir = os.path.join(base_dir, "nnUNet_preprocessed")
    results_dir = os.path.join(base_dir, "nnUNet_results")
    
    for folder in [images_tr, labels_tr, images_ts, preprocessed_dir, results_dir]:
        os.makedirs(folder, exist_ok=True)
        
    print(f"[INFO] nnU-Net v2 Dizin Hiyerarşisi Hazır: {raw_dir}")
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": formatted_dataset_name,
        "dataset_dir": raw_dir,
        "imagesTr": images_tr,
        "labelsTr": labels_tr,
        "imagesTs": images_ts,
        "preprocessed": preprocessed_dir,
        "results": results_dir,
        "base_dir": base_dir
    }


def generate_dataset_json(dataset_dir: str, num_training_cases: int, 
                          labels: Dict[str, int] = None, 
                          file_ending: str = ".nii.gz") -> str:
    """
    nnU-Net v2 otomatik planlama için dataset.json dosyasını çok sınıflı şemayla üretir.
    """
    if labels is None:
        labels = {
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
        
    json_dict = {
        "channel_names": {
            "0": "CT"  # Tek kanallı BT (Hounsfield Unit)
        },
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending
    }
    
    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)
        
    print(f"[SUCCESS] dataset.json üretildi ({num_training_cases} vaka, {len(labels)-1} sınıf): {json_path}")
    return json_path


def auto_scan_and_convert(raw_images_dir: str, raw_labels_dir: Optional[str], 
                          target_paths: Dict[str, str], prefix: str = "hasta") -> int:
    """
    Girdi klasöründeki NIfTI dosyalarını tarar ve nnU-Net v2 formatına (_0000.nii.gz) aktarır.
    """
    if not os.path.exists(raw_images_dir):
        print(f"[ERROR] Ham görüntü klasörü bulunamadı: {raw_images_dir}")
        return 0
        
    image_files = sorted([f for f in os.listdir(raw_images_dir) if f.endswith('.nii.gz') or f.endswith('.nii')])
    print(f"[INFO] Toplam {len(image_files)} adet görüntü dosyası tespit edildi.")
    
    images_tr_dir = target_paths["imagesTr"]
    labels_tr_dir = target_paths["labelsTr"]
    
    count = 0
    for idx, img_file in enumerate(image_files, start=1):
        case_id = f"{prefix}_{idx:03d}"
        
        src_img_path = os.path.join(raw_images_dir, img_file)
        dst_img_path = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
        shutil.copy2(src_img_path, dst_img_path)
        
        if raw_labels_dir and os.path.exists(raw_labels_dir):
            possible_label_names = [img_file, img_file.replace("_0000", ""), img_file.replace("volume", "mask")]
            found_label = None
            for lbl_name in possible_label_names:
                candidate = os.path.join(raw_labels_dir, lbl_name)
                if os.path.exists(candidate):
                    found_label = candidate
                    break
                    
            if found_label:
                dst_lbl_path = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")
                shutil.copy2(found_label, dst_lbl_path)
                
        count += 1
        print(f"  [+] Vaka {case_id} aktarıldı.")
        
    generate_dataset_json(target_paths["dataset_dir"], num_training_cases=count)
    return count


def generate_cloud_training_script(target_paths: Dict[str, str], train_all_folds: bool = True) -> str:
    """
    RunPod / Vast.ai sunucularında nnU-Net preprocessing ve 5-Fold Cross Validation 
    eğitimini başlatan Bash script'ini (run_cloud_training.sh) üretir.
    """
    base_dir = os.path.abspath(target_paths["base_dir"])
    dataset_id = target_paths["dataset_id"]
    
    folds_command = 'for fold in {0..4}; do nnUNetv2_train ' + f'{dataset_id}' + ' 3d_fullres $fold; done' if train_all_folds else f'nnUNetv2_train {dataset_id} 3d_fullres 0'

    script_content = f"""#!/bin/bash
# ==============================================================================
# HepaRECIST-AI: RunPod / Cloud GPU Training Script for nnU-Net v2
# ==============================================================================

export nnUNet_raw="{os.path.join(base_dir, 'nnUNet_raw')}"
export nnUNet_preprocessed="{os.path.join(base_dir, 'nnUNet_preprocessed')}"
export nnUNet_results="{os.path.join(base_dir, 'nnUNet_results')}"

echo "=== nnU-Net v2 Bulut Eğitim Boru Hattı Başlatılıyor ==="
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# 1. Otomatik Planlama ve Ön İşleme (Fingerprint & Preprocessing)
echo "--> Adım 1: Otomatik Planlama & Preprocessing (Dataset {dataset_id:03d})..."
nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity

# Hasta düzeyinde katlar: nnU-Net'in vaka düzeyindeki rastgele bölmesi yerine
# (aynı hastanın t0/t1 serileri aynı katta kalır)
DS_NAME=$(basename $(ls -d "$nnUNet_raw"/Dataset{dataset_id:03d}_*))
if [ -f "$nnUNet_raw/$DS_NAME/splits_final.json" ]; then
  cp "$nnUNet_raw/$DS_NAME/splits_final.json" "$nnUNet_preprocessed/$DS_NAME/splits_final.json"
  echo "--> Hasta düzeyinde splits_final.json kopyalandı."
fi

# 2. 3D Full Resolution Eğitimi Başlat (5-Fold Cross Validation)
echo "--> Adım 2: 3D Full Resolution Eğitimi Başlatılıyor..."
{folds_command}

echo "=== Eğitim Tamamlandı! Model Ağırlıkları nnUNet_results Klasöründe ==="
"""
    script_path = os.path.join(base_dir, "run_cloud_training.sh")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
        
    print(f"[SUCCESS] Bulut eğitim scripti üretildi: {script_path}")
    return script_path


if __name__ == "__main__":
    print("=== HepaRECIST-AI: nnU-Net v2 Veri Hazırlama Modülü ===")
    target_base = os.path.abspath("./nnunet_data")
    env_paths = setup_nnunet_environment(target_base, dataset_id=1, dataset_name="AbdominalTumor")
    generate_cloud_training_script(env_paths)


def _patient_key(case_id: str) -> str:
    """'hasta_012_t1' -> 'hasta_012' (zaman noktası soneki atılır)."""
    parts = case_id.split("_")
    if len(parts) >= 3 and parts[-1].lower().startswith("t"):
        return "_".join(parts[:-1])
    return case_id


def make_patient_level_splits(nnunet_paths: Dict[str, str], case_ids, n_folds: int = 5,
                              test_fraction: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """
    Hasta düzeyinde bölme:
      1. Hastaların test_fraction kadarı ayrı test setine ayrılır; serileri
         imagesTr/labelsTr'den imagesTs/labelsTs'ye taşınır (eğitimde hiç görülmez).
      2. Kalan hastalar n_folds kata bölünür; splits_final.json yazılır
         (nnU-Net v2 formatı: [{"train": [...], "val": [...]}, ...]).
    """
    import json
    import random
    import shutil as _sh

    groups: Dict[str, List[str]] = {}
    for cid in case_ids:
        groups.setdefault(_patient_key(cid), []).append(cid)
    patients = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_test = int(round(len(patients) * test_fraction)) if len(patients) > n_folds else 0
    test_patients = sorted(patients[:n_test])
    train_patients = patients[n_test:]

    raw = nnunet_paths["dataset_dir"]
    labels_ts = os.path.join(raw, "labelsTs")
    os.makedirs(labels_ts, exist_ok=True)
    test_cases = []
    for p in test_patients:
        for cid in groups[p]:
            for src_dir, dst_dir, name in (
                (nnunet_paths["imagesTr"], nnunet_paths["imagesTs"], f"{cid}_0000.nii.gz"),
                (nnunet_paths["labelsTr"], labels_ts, f"{cid}.nii.gz"),
            ):
                src = os.path.join(src_dir, name)
                if os.path.exists(src):
                    _sh.move(src, os.path.join(dst_dir, name))
            test_cases.append(cid)

    k = max(1, min(n_folds, len(train_patients)))
    folds = [train_patients[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        val = sorted(c for p in folds[i] for c in groups[p])
        train = sorted(c for j in range(k) if j != i for p in folds[j] for c in groups[p])
        splits.append({"train": train, "val": val})

    with open(os.path.join(raw, "splits_final.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(raw, "patient_split.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "test_patients": test_patients,
                   "train_patients": sorted(train_patients)}, f, indent=2, ensure_ascii=False)
    pre = os.path.join(nnunet_paths["preprocessed"], nnunet_paths["dataset_name"])
    os.makedirs(pre, exist_ok=True)
    _sh.copy2(os.path.join(raw, "splits_final.json"), os.path.join(pre, "splits_final.json"))

    train_cases = sorted(c for p in train_patients for c in groups[p])
    print(f"[INFO] Hasta düzeyinde bölme: {len(train_patients)} eğitim / {len(test_patients)} test hastası, {k} kat.")
    return {"train_cases": train_cases, "test_cases": sorted(test_cases),
            "test_patients": test_patients, "splits": splits}
