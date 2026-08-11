import os
import sys
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any


def run_segmentation_inference(
    input_nifti_path: str,
    output_mask_path: str,
    model_dir: str = "./models"
) -> Dict[str, Any]:
    """
    nnU-Net v2 3D Lezyon Segmentasyon Çıkarım Motoru.

    Eğitilmiş model ağırlıkları varsa gerçek nnU-Net tahmini yapar.
    Ağırlıklar henüz mevcut değilse (geliştirme/test ortamı) güvenli Mock moduna geçer
    ve pipeline'ın geri kalanını aksatmadan test edilmesine izin verir.

    Çıktı maskesi etiket şeması (nnU-Net v2 standartı):
        0  -> Arka plan (Background)
        1  -> Karaciğer parankimi (Liver parenchyma)
        2  -> Lezyon / Tümör odağı (Lesion / Tumor focus)

    Args:
        input_nifti_path : Normalize edilmiş [0,1] BT NIfTI dosyasının yolu.
        output_mask_path : Üretilen 3D maske NIfTI dosyasının kaydedileceği yol.
        model_dir        : nnU-Net checkpoint_final.pth dosyasını içeren klasör.

    Returns:
        detected_lesions_count  : Tespit edilen lezyon sayısı.
        total_volume_mm3        : Toplam lezyon hacmi (mm³).
        estimated_diameter_mm   : Küresel yaklaşımla hesaplanan çap (mm).
        is_mock                 : True ise Mock modunda çalışılmıştır.
        output_mask_path        : Üretilen maske dosyasının yolu.
        preview_image_path      : Oluşturulan 2D kesit PNG görselinin yolu (boş string ise üretilemedi).
    """
    if not os.path.exists(input_nifti_path):
        raise FileNotFoundError(f"Girdi NIfTI dosyası bulunamadı: {input_nifti_path}")

    output_dir = os.path.dirname(output_mask_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ./models/ klasöründe .pth veya .pt uzantılı ağırlık dosyası ara
    model_weights_exist = (
        os.path.exists(model_dir) and
        any(f.endswith('.pth') or f.endswith('.pt') for f in os.listdir(model_dir))
    )

    is_mock = True
    if model_weights_exist:
        print(f"[INFO] Eğitilmiş nnU-Net ağırlıkları bulundu: {model_dir}. Gerçek AI çıkarımı başlatılıyor...")
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=device
            )
            predictor.initialize_from_trained_model_folder(
                model_dir,
                use_folds=(0,),
                checkpoint_name='checkpoint_final.pth'
            )
            predictor.predict_from_files(
                [[input_nifti_path]],
                [output_mask_path],
                save_probabilities=False
            )
            print(f"[SUCCESS] Gerçek nnU-Net AI Segmentasyonu tamamlandı -> {output_mask_path}")
            is_mock = False
        except Exception as exc:
            print(f"[WARNING] nnU-Net çağrısı başarısız ({exc}). Mock moduna geçiliyor...")
            _generate_mock_mask(input_nifti_path, output_mask_path)
    else:
        print(f"[INFO] Model ağırlıkları bulunamadı ({model_dir}). Güvenli Mock Modu aktif.")
        _generate_mock_mask(input_nifti_path, output_mask_path)

    result = _calculate_mask_metrics(output_mask_path)
    result["is_mock"] = is_mock
    result["output_mask_path"] = output_mask_path

    # Segmentasyon tamamlanınca 2D BT kesit görselini otomatik üret
    try:
        from visualizer import generate_lesion_visualization
        preview_png = output_mask_path.replace(".nii.gz", "_preview.png")
        generate_lesion_visualization(
            ct_nifti_path=input_nifti_path,
            mask_nifti_path=output_mask_path,
            output_png_path=preview_png
        )
        result["preview_image_path"] = preview_png
    except Exception as exc:
        print(f"[WARNING] 2D kesit görseli üretilemedi: {exc}")
        result["preview_image_path"] = ""

    return result


def _generate_mock_mask(input_nifti_path: str, output_mask_path: str) -> None:
    """
    Model henüz eğitilmemişken test ortamı için 3D sahte lezyon maskesi üretir.
    Görüntünün tam ortasına 14x14x14 voxel boyutunda bir lezyon kümesi (label=2) yerleştirir.
    Bu, pipeline'ın segmentasyon sonraki tüm adımlarının (radiomics, matching, rapor)
    bozulmadan test edilmesini sağlar.
    """
    image = sitk.ReadImage(input_nifti_path)
    volume = sitk.GetArrayFromImage(image)

    mask_volume = np.zeros_like(volume, dtype=np.uint8)
    z_mid, y_mid, x_mid = [dim // 2 for dim in volume.shape]
    # Merkeze 14x14x14 voxel sahte lezyon kümesi
    mask_volume[z_mid-7:z_mid+7, y_mid-7:y_mid+7, x_mid-7:x_mid+7] = 2

    mask_image = sitk.GetImageFromArray(mask_volume)
    mask_image.CopyInformation(image)
    sitk.WriteImage(mask_image, output_mask_path)


def _calculate_mask_metrics(mask_path: str) -> Dict[str, Any]:
    """
    Üretilen 3D maske dosyasını okuyarak lezyon hacmi ve küresel çap tahminini hesaplar.

    Çap tahmini yöntemi:
        Hacim formülü V = (4/3) * pi * r^3 tersine çevrilerek r = (3V / 4pi)^(1/3)
        ve d = 2r bulunur. Bu, 2D Feret çapının küresel yaklaşımıdır.
    """
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()  # (x_mm, y_mm, z_mm)

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    lesion_voxels = int(np.sum(mask_arr == 2))  # Sadece label=2 (Lezyon) sayılır
    total_volume_mm3 = float(lesion_voxels * voxel_volume_mm3)

    if total_volume_mm3 > 0:
        estimated_diameter_mm = 2.0 * ((3.0 * total_volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        estimated_diameter_mm = 0.0

    return {
        "detected_lesions_count": 1 if lesion_voxels > 0 else 0,
        "total_volume_mm3": round(total_volume_mm3, 2),
        "estimated_diameter_mm": round(estimated_diameter_mm, 2)
    }


if __name__ == "__main__":
    print("=== nnU-Net Segmentasyon Çıkarım Modülü (inference.py) ===")
