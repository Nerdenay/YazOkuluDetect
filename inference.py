"""
inference.py
============
HepaRECIST-AI - Faz 2: nnU-Net v2 Çok Sınıflı Batın Segmentasyon Çıkarım Motoru
- 8-Sınıflı Batın Organları + Lezyon (Label 8) Tahmini
- Eğitilmiş model varsa gerçek nnU-Net v2 Sliding-Window çıkarımı
- Model yoksa veya çıkarım başarısızsa HATA verir (sahte lezyon üretmez).
  Test amaçlı sahte maske yalnızca allow_mock=True ya da HEPARECIST_ALLOW_MOCK=1
  ile açıkça istenirse üretilir ve sonuçta is_mock=True olarak işaretlenir.
- nnU-Net v2 truncated dosya isimlendirmesi ve RAM/VRAM dengeli tahmin mimarisi
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, List, Optional

from lesion_geometry import extract_lesions


class ModelUnavailableError(RuntimeError):
    """Gerçek model çıkarımı yapılamadığında fırlatılır (sahte sonuç üretilmez)."""


# ── ANA ÇIKARIM FONKSİYONU ───────────────────────────────────────────────────

def run_segmentation_inference(
    input_nifti_path: str,
    output_mask_path: str,
    model_dir: str = "./models",
    lesion_label_id: int = 8,
    allow_mock: Optional[bool] = None,
    slice_thickness_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    nnU-Net v2 ile 3D çok sınıflı batın segmentasyonu çıkarımı yapar.
    Model yoksa veya çıkarım başarısız olursa ModelUnavailableError fırlatır.
    """
    if allow_mock is None:
        allow_mock = os.environ.get("HEPARECIST_ALLOW_MOCK", "0") == "1"
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
            if not allow_mock:
                raise ModelUnavailableError(f"nnU-Net çıkarımı başarısız: {exc}") from exc
            print(f"[TEST] nnU-Net çıkarımı başarısız ({exc}); allow_mock açık, SAHTE maske üretiliyor.")
            _generate_mock_mask(input_nifti_path, output_mask_path, lesion_label_id=lesion_label_id)
    else:
        if not allow_mock:
            raise ModelUnavailableError(
                f"Model ağırlıkları bulunamadı ({model_dir}). Gerçek hasta verisi için eğitilmiş "
                "nnU-Net modeli gereklidir. Test için allow_mock=True / HEPARECIST_ALLOW_MOCK=1 kullanın.")
        print(f"[TEST] Model ağırlıkları yok ({model_dir}); allow_mock açık, SAHTE maske üretiliyor.")
        _generate_mock_mask(input_nifti_path, output_mask_path, lesion_label_id=lesion_label_id)

    # Üretilen maskenin lezyon analizini yap
    result = _calculate_mask_metrics(output_mask_path, lesion_label_id=lesion_label_id,
                                     slice_thickness_mm=slice_thickness_mm)
    result["is_mock"] = is_mock
    result["output_mask_path"] = output_mask_path

    # Önizleme 2D PNG görseli oluşturma (Varsa visualizer modülü)
    try:
        from visualizer import generate_lesion_visualization
        base_name = output_mask_path
        for ext in [".nii.gz", ".nii"]:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        preview_png = base_name + "_preview.png"
        generate_lesion_visualization(
            ct_nifti_path=input_nifti_path,
            mask_nifti_path=output_mask_path,
            output_png_path=preview_png,
            lesion_label_id=lesion_label_id
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


def _calculate_mask_metrics(mask_path: str, lesion_label_id: int = 8,
                            slice_thickness_mm: Optional[float] = None) -> Dict[str, Any]:
    """
    3D maskeden lezyon odaklarını ortak ölçüm modülüyle (lesion_geometry) çıkarır.
    """
    mask_img = sitk.ReadImage(mask_path)
    lesions, _ = extract_lesions(mask_img, lesion_label_id, slice_thickness_mm)

    lesion_details = [{
        "lesion_id": l["id"],
        "organ": l["organ"],
        "volume_mm3": l["volume_mm3"],
        "axial_diameter_mm": l["diameter_mm"],
        "is_measurable": l["is_measurable"],
        "is_target": l["is_measurable"],
    } for l in lesions]
    primary = max((l["diameter_mm"] for l in lesions), default=0.0)

    return {
        "detected_lesions_count": len(lesion_details),
        "total_volume_mm3": round(sum(l["volume_mm3"] for l in lesions), 2),
        "primary_lesion_diameter_mm": round(primary, 2),
        "estimated_diameter_mm": round(primary, 2),  # main.py uyumluluğu için
        "lesions": lesion_details
    }


if __name__ == "__main__":
    print("=== HepaRECIST-AI: Segmentasyon Çıkarım Modülü (inference.py) ===")
