"""
preprocess.py
=============
HepaRECIST-AI - Faz 1: DICOM Ön İşleme ve Standardizasyon Modülü
- 1. Tercih: dicom2nifti (standart dönüşüm + RAS reorientation)
- 2. Tercih (Fallback): SimpleITK ImageSeriesReader (PACS uyumlu, yönelim ve koordinatları tam korur)
- Ham HU değerlerini korur (TotalSegmentator ve nnU-Net için zorunludur).
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk

try:
    import dicom2nifti
    import dicom2nifti.settings as d2n_settings
    # Çok katı DICOM geometri kontrollerini esnet (PACS serilerinde çökmemesi için)
    d2n_settings.disable_validate_slice_increment()
except ImportError:
    dicom2nifti = None


def convert_dicom_to_nifti(dicom_dir: str, output_dir: str, output_filename: str) -> str:
    """
    DICOM serisini koordinat, yönelim ve spacing kaybı olmadan NIfTI (.nii.gz) formatına çevirir.
    Önce dicom2nifti dener; hata verirse SimpleITK GDCM motoruna geçer.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    print(f"[INFO] DICOM dönüştürülüyor: {dicom_dir} -> {output_path}")

    # 1. Yöntem: dicom2nifti
    if dicom2nifti is not None:
        temp_d2n_dir = os.path.join(output_dir, "_tmp_d2n")
        os.makedirs(temp_d2n_dir, exist_ok=True)
        try:
            dicom2nifti.convert_directory(dicom_dir, temp_d2n_dir, compression=True, reorient=True)
            generated = [f for f in os.listdir(temp_d2n_dir) if f.endswith('.nii.gz')]
            
            if generated:
                # Birden fazla seri üretildiyse en büyük boyutluyu (ana aksiyal BT) seç
                generated.sort(key=lambda f: os.path.getsize(os.path.join(temp_d2n_dir, f)), reverse=True)
                best_file = os.path.join(temp_d2n_dir, generated[0])
                
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(best_file, output_path)
                shutil.rmtree(temp_d2n_dir, ignore_errors=True)

                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print("[SUCCESS] dicom2nifti ile başarıyla dönüştürüldü.")
                    return output_path
        except Exception as e:
            print(f"[INFO] dicom2nifti uyarı/hata: {e}. SimpleITK fallback deneniyor...")
        finally:
            shutil.rmtree(temp_d2n_dir, ignore_errors=True)

    # 2. Yöntem: SimpleITK ImageSeriesReader (PACS ve uzantısız dosyalar için en sağlam motor)
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        
        # Eğer GetGDCMSeriesFileNames dosyaları bulamazsa klasördeki tüm dosyaları zorla ver
        if not dicom_names:
            all_files = [
                os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                if not f.startswith(('.', '_')) and f.upper() not in ["DICOMDIR"]
                and not f.endswith(('.txt', '.xml', '.pdf', '.jpg', '.png'))
            ]
            if not all_files:
                raise RuntimeError(f"Klasörde işlenebilecek DICOM dosyası bulunamadı: {dicom_dir}")
            dicom_names = tuple(all_files)

        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        image = reader.Execute()

        # SimpleITK LPS koordinat sistemindedir; DICOM yönelimini tam korur
        sitk.WriteImage(image, output_path)
        print(f"[SUCCESS] SimpleITK ile dönüştürüldü (Boyut: {image.GetSize()}, Spacing: {image.GetSpacing()})")
        return output_path

    except Exception as e:
        print(f"[FATAL] SimpleITK dönüşümü de başarısız oldu: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def resample_image(image_path: str, output_path: str, new_spacing: tuple = (1.0, 1.0, 1.0), is_label: bool = False) -> str:
    """
    Görüntüyü izotropik spacing'e (örn: 1.0x1.0x1.0 mm) resample eder.
    nnU-Net kendi resampling'ini yaptığı için bu işlem genellikle Faz 4 (Longitudinal Registration)
    öncesinde T0-T1 eşitlemesi için kullanılır.
    """
    print(f"[INFO] Resampling uygulanıyor: Spacing -> {new_spacing}")
    try:
        image = sitk.ReadImage(image_path)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        if np.allclose(original_spacing, new_spacing, atol=1e-3):
            if image_path != output_path:
                sitk.WriteImage(image, output_path)
            return output_path

        new_size = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(3)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

        resampled_image = resample.Execute(image)
        sitk.WriteImage(resampled_image, output_path)
        return output_path
    except Exception as e:
        print(f"[ERROR] Resampling hatası: {e}")
        raise


def apply_windowing_for_visualization(nifti_path: str, output_path: str, wl: int = 40, ww: int = 150,
                                      window_level: int = None, window_width: int = None) -> str:
    """
    SADECE UI / Görselleştirme amaçlıdır! (TotalSegmentator veya nnU-Net eğitimine SOKULMAZ).
    Yumuşak doku penceresi uygulayıp [0, 1] aralığına çeker.
    """
    if window_level is not None:
        wl = window_level
    if window_width is not None:
        ww = window_width

    image = sitk.ReadImage(nifti_path)
    volume = sitk.GetArrayFromImage(image)

    min_hu = wl - (ww / 2.0)
    max_hu = wl + (ww / 2.0)

    clipped = np.clip(volume, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)

    proc_img = sitk.GetImageFromArray(normalized.astype(np.float32))
    proc_img.CopyInformation(image)
    sitk.WriteImage(proc_img, output_path)
    return output_path

# Geriye dönük uyumluluk takma adı (main.py ve pipeline.py için)
apply_windowing = apply_windowing_for_visualization


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Kullanım: python preprocess.py <dicom_klasor> <cikti_klasor> <dosya_adi.nii.gz>")
    else:
        # Ham HU değerleriyle NIfTI üretir (Doğrudan TotalSegmentator & nnU-Net uyumlu)
        convert_dicom_to_nifti(sys.argv[1], sys.argv[2], sys.argv[3])
