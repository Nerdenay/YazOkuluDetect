import os
import sys
import dicom2nifti
import SimpleITK as sitk
import numpy as np

def convert_dicom_to_nifti(dicom_dir: str, output_dir: str, output_filename: str) -> str:
    """
    Belirtilen DICOM klasöründeki kesitleri okur ve tek bir .nii.gz (NIfTI) dosyasına dönüştürür.
    Sectra PACS ve farklı DICOM formatları için hem dicom2nifti hem de SimpleITK motorunu destekler.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, output_filename)
    print(f"[INFO] DICOM serisi dönüştürülüyor: {dicom_dir} -> {output_path}")

    # 1. Yöntem: dicom2nifti dene
    try:
        dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=True)
        generated_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz') and f != output_filename]
        if generated_files:
            temp_path = os.path.join(output_dir, generated_files[0])
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            
        if os.path.exists(output_path):
            print("[SUCCESS] DICOM -> NIfTI dönüşümü tamamlandı (dicom2nifti).")
            return output_path
    except Exception as e:
        print(f"[INFO] dicom2nifti deneniyor, SimpleITK fallback'e geçiliyor... ({e})")

    # 2. Yöntem: SimpleITK ImageSeriesReader (PACS ve uzantısız dosyalar için en sağlamı)
    try:
        reader = sitk.ImageSeriesReader()
        
        # Klasörün içindeki ve alt klasörlerdeki serileri tara
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
        # Eğer doğrudan bulunamadıysa alt klasörlere bak
        target_dir = dicom_dir
        if not series_ids:
            for root, dirs, files in os.walk(dicom_dir):
                ids = reader.GetGDCMSeriesIDs(root)
                if ids:
                    series_ids = ids
                    target_dir = root
                    break

        if not series_ids:
            raise RuntimeError(f"DICOM serisi bulunamadı: {dicom_dir}")

        # En çok kesite sahip olan seriyi seç (aksiyal ana hacim)
        best_series = None
        max_files = 0
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(target_dir, sid)
            if len(files) > max_files:
                max_files = len(files)
                best_series = sid

        if not best_series or max_files == 0:
            raise RuntimeError(f"Geçerli DICOM kesiti bulunamadı: {dicom_dir}")

        dicom_files = reader.GetGDCMSeriesFileNames(target_dir, best_series)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        sitk.WriteImage(image, output_path)
        print(f"[SUCCESS] DICOM -> NIfTI dönüşümü tamamlandı (SimpleITK: {max_files} kesit).")
        return output_path

    except Exception as e:
        print(f"[ERROR] Dönüşüm başarısız oldu: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def apply_windowing(nifti_path: str, output_path: str, window_level: int = 40, window_width: int = 150) -> str:
    """
    NIfTI görüntüsüne Hounsfield Unit (HU) pencereleme (clipping) uygular 
    ve veriyi [0, 1] aralığına normalize eder.
    
    Karaciğer / Yumuşak doku için varsayılan: WL=40, WW=150 (Aralık: -35 ile 115 HU)
    
    Args:
        nifti_path (str): Giriş NIfTI dosyasının yolu.
        output_path (str): Çıkış NIfTI dosyasının yolu.
        window_level (int): Pencere seviyesi (Center).
        window_width (int): Pencere genişliği (Width).
        
    Returns:
        str: İşlenmiş NIfTI dosyasının yolu.
    """
    print(f"[INFO] HU Pencereleme uygulanıyor (WL: {window_level}, WW: {window_width})...")
    
    try:
        # SimpleITK ile görüntüyü oku
        image = sitk.ReadImage(nifti_path)
        
        volume = sitk.GetArrayFromImage(image)
        
        min_hu = window_level - (window_width / 2)
        max_hu = window_level + (window_width / 2)
        
        clipped_volume = np.clip(volume, min_hu, max_hu)
        
        normalized_volume = (clipped_volume - min_hu) / (max_hu - min_hu)
        
        processed_image = sitk.GetImageFromArray(normalized_volume.astype(np.float32))
        
        processed_image.CopyInformation(image)
        
        sitk.WriteImage(processed_image, output_path)
        print(f"[SUCCESS] Pencereleme ve normalizasyon tamamlandı: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Pencereleme sırasında hata oluştu: {str(e)}")
        raise

def resample_image(image_path: str, output_path: str, new_spacing: tuple = (1.0, 1.0, 1.0), is_label: bool = False) -> str:
    """
    Bir görüntüyü (veya maskeyi) belirtilen hedef spacing (voxel boyutu) değerine yeniden örnekler (resample).
    
    Args:
        image_path (str): Giriş NIfTI görüntüsünün yolu.
        output_path (str): Çıkış NIfTI görüntüsünün yolu.
        new_spacing (tuple): Hedef spacing (x, y, z) milimetre cinsinden. Varsayılan (1.0, 1.0, 1.0).
        is_label (bool): Görüntü maske/etiket ise True olmalıdır. Bu durumda Nearest Neighbor interpolasyonu kullanılır.
        
    Returns:
        str: Yeniden örneklenmiş görüntünün kaydedildiği yol.
    """
    print(f"[INFO] Resampling uygulanıyor: {image_path} -> Spacing: {new_spacing}")
    
    try:
        image = sitk.ReadImage(image_path)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Eğer orijinal spacing ile hedef spacing zaten aynıysa işlem yapmadan kaydet/kopyala
        if np.allclose(original_spacing, new_spacing):
            print("[INFO] Görüntü zaten hedef spacing değerine sahip. İşlem atlanıyor.")
            sitk.WriteImage(image, output_path)
            return output_path
            
        # Yeni boyutu hesapla: yeni_boyut = eski_boyut * (eski_spacing / yeni_spacing)
        new_size = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(3)
        ]
        
        # Resampling filtresini yapılandır
        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())
        
        # Interpolatör seçimi (Görüntü ve maske için farklı filtreler kullanılır)
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)  # Maske için
        else:
            resample.SetInterpolator(sitk.sitkLinear)  # CT görüntüsü için
            
        # Filtreyi çalıştır ve kaydet
        resampled_image = resample.Execute(image)
        sitk.WriteImage(resampled_image, output_path)
        
        print(f"[SUCCESS] Resampling tamamlandı. Boyut: {original_size} -> {new_size}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Resampling sırasında hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Kullanım: python preprocess.py <dicom_klasor_yolu> <cikti_klasor_yolu> <cikti_dosya_adi>")
        print("Örnek: python preprocess.py ./sample_dicom ./output patient_baseline.nii.gz")
    else:
        dicom_in = sys.argv[1]
        out_dir = sys.argv[2]
        filename = sys.argv[3]
        
        try:
            # 1. Adım: DICOM -> NIfTI
            raw_nifti = convert_dicom_to_nifti(dicom_in, out_dir, "raw_" + filename)
            
            # 2. Adım: HU Windowing & Normalizasyon
            windowed_nifti = os.path.join(out_dir, "windowed_" + filename)
            apply_windowing(raw_nifti, windowed_nifti)
            
            # 3. Adım: Resampling (Hedef: 1.0mm x 1.0mm x 1.0mm)
            final_nifti = os.path.join(out_dir, filename)
            resample_image(windowed_nifti, final_nifti, new_spacing=(1.0, 1.0, 1.0), is_label=False)
            
            # Geçici dosyaları temizle
            os.remove(raw_nifti)
            os.remove(windowed_nifti)
            print("[INFO] İşlem başarıyla sonlandı. Son çıktı:", final_nifti)
        except Exception as ex:
            print(f"[FATAL] Hata: {ex}")
