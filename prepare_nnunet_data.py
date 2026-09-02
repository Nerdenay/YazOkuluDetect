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
from typing import List, Dict, Optional


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
echo "nnUNet_preprocessed: $nnUN_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# 1. Otomatik Planlama ve Ön İşleme (Fingerprint & Preprocessing)
echo "--> Adım 1: Otomatik Planlama & Preprocessing (Dataset {dataset_id:03d})..."
nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity

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
