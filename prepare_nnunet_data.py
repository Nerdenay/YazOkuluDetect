import os
import sys
import shutil
import json
from typing import List, Dict, Optional

def setup_nnunet_environment(base_dir: str, dataset_id: int = 1, dataset_name: str = "LiverLesion") -> Dict[str, str]:
    """
    nnU-Net v2 için gerekli tüm ortam dizinlerini ve DatasetXXX_Name klasör hiyerarşisini kurar.
    
    Örnek: nnUNet_raw/Dataset001_LiverLesion/(imagesTr, labelsTr, imagesTs)
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
    nnU-Net v2'nin otomatik planlama yapabilmesi için dataset.json dosyasını dinamik üretir.
    """
    if labels is None:
        labels = {
            "background": 0,
            "liver": 1,
            "lesion": 2
        }
        
    json_dict = {
        "channel_names": {
            "0": "CT"  # Tek kanallı BT taraması
        },
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending
    }
    
    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)
        
    print(f"[SUCCESS] dataset.json üretildi ({num_training_cases} vaka): {json_path}")
    return json_path

def auto_scan_and_convert(raw_images_dir: str, raw_labels_dir: Optional[str], 
                           target_paths: Dict[str, str], prefix: str = "case") -> int:
    """
    Girdi klasöründeki TÜM NIfTI dosyalarını (sayıdan bağımsız: 3, 50, 500+) 
    otomatik tarar ve nnU-Net v2 formatına (prefix_XXX_0000.nii.gz) aktarır.
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
        
        # Kaynak yolları
        src_img_path = os.path.join(raw_images_dir, img_file)
        
        # Hedef yolları (nnU-Net v2 standardı: _0000 takısı zorunludur)
        dst_img_path = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
        shutil.copy2(src_img_path, dst_img_path)
        
        # Etiket maskesi varsa eşleştir
        if raw_labels_dir and os.path.exists(raw_labels_dir):
            # Dosya ismiyle aynı veya benzeyen maske ara
            possible_label_names = [img_file, img_file.replace("volume", "segmentation"), img_file.replace("image", "mask")]
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
        print(f"  [+] Vaka {case_id} aktarıldı ({img_file} -> {os.path.basename(dst_img_path)})")
        
    generate_dataset_json(target_paths["dataset_dir"], num_training_cases=count)
    return count

def generate_cloud_training_script(target_paths: Dict[str, str]) -> str:
    """
    RunPod / Vast.ai / Linux GPU sunucularında tek tıkla 
    eğitim ve preprocessing başlatacak Bash script'ini (run_cloud_training.sh) üretir.
    """
    base_dir = os.path.abspath(target_paths["base_dir"])
    dataset_id = target_paths["dataset_id"]
    
    script_content = f"""#!/bin/bash
# ==============================================================================
# RunPod / Vast.ai Cloud GPU Training Script for nnU-Net v2
# ==============================================================================

# 1. nnU-Net Ortam Değişkenlerini Tanımla
export nnUNet_raw="{os.path.join(base_dir, 'nnUNet_raw')}"
export nnUNet_preprocessed="{os.path.join(base_dir, 'nnUNet_preprocessed')}"
export nnUNet_results="{os.path.join(base_dir, 'nnUNet_results')}"

echo "=== nnU-Net v2 Bulut Eğitim Boru Hattı Başlatılıyor ==="
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# 2. Otomatik Planlama ve Ön İşleme (Fingerprint & Preprocessing)
echo "--> Step 1: Otomatik Planlama & Preprocessing (Dataset {dataset_id:03d})..."
nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity

# 3. 5-Fold Cross Validation Eğitimi Başlat (3d_fullres)
echo "--> Step 2: 3D Full Resolution Eğitimi Başlatılıyor (Fold 0)..."
nnUNetv2_train {dataset_id} 3d_fullres 0

echo "=== Eğitim Tamamlandı! Sonuçlar nnUNet_results Klasöründe ==="
"""
    script_path = os.path.join(base_dir, "run_cloud_training.sh")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
        
    print(f"[SUCCESS] Bulut (RunPod) eğitim scripti üretildi: {script_path}")
    return script_path

if __name__ == "__main__":
    print("=== Genel ve Ölçeklenebilir nnU-Net v2 Veri Hazırlama Modülü ===")
    
    # Varsayılan çalışma dizini
    target_base = os.path.abspath("./nnunet_data")
    env_paths = setup_nnunet_environment(target_base, dataset_id=1, dataset_name="LiverLesion")
    
    # Örnek Kullanım Açıklaması
    print("\n[KULLANIM] Herhangi bir veri setini işlemek için:")
    print("  from prepare_nnunet_data import setup_nnunet_environment, auto_scan_and_convert, generate_cloud_training_script")
    print("  paths = setup_nnunet_environment('./nnunet_data', dataset_id=1, dataset_name='LiverLesion')")
    print("  auto_scan_and_convert(raw_images_dir='./my_raw_images', raw_labels_dir='./my_raw_labels', target_paths=paths)")
    print("  generate_cloud_training_script(paths)")
