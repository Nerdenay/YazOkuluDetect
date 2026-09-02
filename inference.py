"""
inference.py
============
HepaRECIST-AI - Faz 2: nnU-Net v2 Çok Sınıflı Batın Segmentasyon Çıkarım Motoru
- 8-Sınıflı Batın Organları + Lezyon (Label 8) Tahmini
- Eğitilmiş model varsa gerçek nnU-Net v2 Sliding-Window çıkarımı
- Ağırlık yoksa veya GPU yetersizse pipeline'ı bozmayan Güvenli Mock Modu
- nnU-Net v2 truncated dosya isimlendirmesi ve RAM/VRAM dengeli tahmin mimarisi
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Dict, Any, List


# ── ANA ÇIKARIM FONKSİYONU ───────────────────────────────────────────────────

def run_segmentation_inference(
    input_nifti_path: str,
    output_mask_path: str,
    model_dir: str = "./models",
    lesion_label_id: int = 8
) -> Dict[str, Any]:
    """
    nnU-Net v2 ile 3D çok sınıflı batın segmentasyonu çıkarımı yapar.
    Ağırlıklar yoksa otomatik olarak Mock moduna geçer.
    """
    if not os.path.exists(input_nifti_path):
        raise FileNotFoundError(f"Girdi NIfTI dosyası bulunamadı: {input_nifti_path}")

    output_dir = os.path.dirname(output_mask_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ./models/ klasöründe checkpoint_final.pth veya .pth ara
    has_weights = False
    if os.path.exists(model_dir):
        for root, _, files in os.walk(model_dir):
            if any(f.endswith('.pth') or f.endswith('.pt') for f in files):
                has_weights = True
                break

    is_mock = True

    if has_weights:
        print(f"[INFO] nnU-Net model ağırlıkları bulundu: {model_dir}. AI çıkarımı başlatılıyor...")
        try:
            import torch
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # VRAM Taşmasını (OOM) önlemek için perform_everything_on_device=False yapılır
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,  # Çıkarım hızını 4 kat artırır
                perform_everything_on_device=False,
                device=device
            )
            predictor.initialize_from_trained_model_folder(
                model_dir,
                use_folds=(0,),
                checkpoint_name='checkpoint_final.pth'
            )

            # nnU-Net v2 kuralı: Çıktı dosya adı uzantısız (truncated) olmalıdır!
            truncated_output = output_mask_path
            for ext in [".nii.gz", ".nii"]:
                if truncated_output.endswith(ext):
                    truncated_output = truncated_output[:-len(ext)]

            predictor.predict_from_files(
                [[input_nifti_path]],
                [truncated_output],
                save_probabilities=False
            )

            if os.path.exists(output_mask_path) and os.path.getsize(output_mask_path) > 1000:
                print(f"[SUCCESS] Gerçek nnU-Net segmentasyonu tamamlandı: {output_mask_path}")
                is_mock = False
            else:
                raise RuntimeError("Maske dosyası nnU-Net tarafından üretilemedi.")

        except Exception as exc:
            print(f"[WARNING] nnU-Net çıkarımı başarısız ({exc}). Güvenli Mock Modu devreye giriyor...")
            _generate_mock_mask(input_nifti_path, output_mask_path, lesion_label_id=lesion_label_id)
    else:
        print(f"[INFO] Model ağırlıkları bulunamadı ({model_dir}). Test için Mock Modu aktif.")
        _generate_mock_mask(input_nifti_path, output_mask_path, lesion_label_id=lesion_label_id)

    # Üretilen maskenin lezyon analizini yap
    result = _calculate_mask_metrics(output_mask_path, lesion_label_id=lesion_label_id)
    result["is_mock"] = is_mock
    result["output_mask_path"] = output_mask_path

    # Önizleme 2D PNG görseli oluşturma (Varsa visualizer modülü)
    try:
        from visualizer import generate_lesion_visualization
        preview_png = output_mask_path.replace(".nii.gz", "_preview.png")
        generate_lesion_visualization(
            ct_nifti_path=input_nifti_path,
            mask_nifti_path=output_mask_path,
            output_png_path=preview_png
        )
        result["preview_image_path"] = preview_png
    except Exception:
        result["preview_image_path"] = ""

    return result


# ── YARDIMCI VE MOCK FONKSİYONLARI ──────────────────────────────────────────

def _generate_mock_mask(input_nifti_path: str, output_mask_path: str, lesion_label_id: int = 8) -> None:
    """
    Test ortamı için sahte karaciğer (Label 1) ve lezyon (Label 8) kümesi oluşturur.
    Böylece tüm boru hattı (Hungarian eşleştirme, RECIST SOD hesabı) eksiksiz test edilebilir.
    """
    image = sitk.ReadImage(input_nifti_path)
    volume = sitk.GetArrayFromImage(image)

    mask_volume = np.zeros_like(volume, dtype=np.uint8)
    z_mid, y_mid, x_mid = [dim // 2 for dim in volume.shape]

    # Sahte Karaciğer Parankimi (Label 1)
    mask_volume[max(0, z_mid-30):min(volume.shape[0], z_mid+30),
                max(0, y_mid-50):min(volume.shape[1], y_mid+50),
                max(0, x_mid-60):min(volume.shape[2], x_mid+60)] = 1

    # Karaciğer içine 16x16x16 mm boyutunda Sahte Malign Lezyon (Label 8)
    mask_volume[z_mid-8:z_mid+8, y_mid-8:y_mid+8, x_mid-8:x_mid+8] = lesion_label_id

    mask_image = sitk.GetImageFromArray(mask_volume)
    mask_image.CopyInformation(image)
    sitk.WriteImage(mask_image, output_mask_path)


def _calculate_mask_metrics(mask_path: str, lesion_label_id: int = 8) -> Dict[str, Any]:
    """
    3D maskeden her bir lezyon odağını bağımsız olarak ayıklar,
    toplam lezyon sayısını ve RECIST 1.1 hedef lezyon çapını hesaplar.
    """
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()

    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    # Eğer belirtilen lezyon etiketi maskede yoksa alternatifleri kontrol et (2 veya 1)
    unique_vals = set(np.unique(mask_arr))
    if lesion_label_id not in unique_vals:
        for alt_id in [2, 1]:
            if alt_id in unique_vals:
                lesion_label_id = alt_id
                break

    binary_lesions = (mask_arr == lesion_label_id)

    labeled_mask, num_features = ndimage.label(binary_lesions)
    lesion_details = []
    total_volume_mm3 = 0.0

    for r_id in range(1, num_features + 1):
        voxels = (labeled_mask == r_id)
        v_count = int(np.sum(voxels))
        if v_count < 15:
            continue

        vol = v_count * voxel_vol
        total_volume_mm3 += vol

        # Aksiyel düzlemde maksimum çap hesabı (RECIST 1.1)
        z_slices = np.unique(np.where(voxels)[0])
        max_d = 0.0
        for z in z_slices:
            slice_2d = voxels[z, :, :]
            y_pts, x_pts = np.where(slice_2d)
            if len(x_pts) >= 2:
                pts = np.column_stack((x_pts * spacing[0], y_pts * spacing[1]))
                if len(pts) > 80:
                    pts = pts[::max(1, len(pts)//40)]
                diffs = pts[:, None, :] - pts[None, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=-1))
                max_d = max(max_d, float(np.max(dists)))

        lesion_details.append({
            "lesion_id": r_id,
            "volume_mm3": round(vol, 2),
            "axial_diameter_mm": round(max_d, 2),
            "is_target": max_d >= 10.0
        })

    primary_diameter = max([l["axial_diameter_mm"] for l in lesion_details], default=0.0)

    return {
        "detected_lesions_count": len(lesion_details),
        "total_volume_mm3": round(total_volume_mm3, 2),
        "primary_lesion_diameter_mm": round(primary_diameter, 2),
        "estimated_diameter_mm": round(primary_diameter, 2),  # main.py uyumluluğu için
        "lesions": lesion_details
    }


if __name__ == "__main__":
    print("=== HepaRECIST-AI: Segmentasyon Çıkarım Modülü (inference.py) ===")
