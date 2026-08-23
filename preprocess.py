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

    # 1. Yöntem: dicom2nifti (en basit, bazen birden fazla dosya üretir)
    try:
        dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=True)
        
        # dicom2nifti birden fazla seri üretebilir → EN BÜYÜK dosyayı (ana CT) al
        generated_files = [
            f for f in os.listdir(output_dir)
            if f.endswith('.nii.gz') and f != output_filename
        ]
        if generated_files:
            # Boyuta göre sırala — en büyük = ana aksiyal CT serisi
            generated_files.sort(
                key=lambda f: os.path.getsize(os.path.join(output_dir, f)),
                reverse=True
            )
            best_path = os.path.join(output_dir, generated_files[0])
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(best_path, output_path)
            
            # Geri kalan küçük serileri temizle
            for extra_file in generated_files[1:]:
                extra_path = os.path.join(output_dir, extra_file)
                if os.path.exists(extra_path):
                    os.remove(extra_path)
                    
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("[SUCCESS] DICOM -> NIfTI dönüşümü tamamlandı (dicom2nifti).")
            return output_path
            
    except Exception as e:
        print(f"[INFO] dicom2nifti başarısız, pydicom motoruna geçiliyor... ({type(e).__name__})")
        for f in os.listdir(output_dir):
            if f.endswith('.nii.gz') and f != output_filename:
                try:
                    os.remove(os.path.join(output_dir, f))
                except Exception:
                    pass

    # 2. Yöntem: pydicom + nibabel (uzantısız Sectra dosyaları dahil)
    try:
        import pydicom
        import nibabel as nib

        # Tüm DICOM dosyalarını tara — ImagePositionPatient ZORUNLU DEĞİL
        all_candidates = []
        for root, dirs, files in os.walk(dicom_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.upper() in ["DICOMDIR", ".DS_STORE"] or f.endswith((".txt", ".xml", ".pdf", ".jpg")):
                    continue
                fpath = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                    # En az PixelData olduğunu doğrulamak için Rows/Columns kontrolü
                    if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "SeriesInstanceUID"):
                        all_candidates.append((ds.SeriesInstanceUID, fpath, ds))
                except Exception:
                    continue

        if not all_candidates:
            raise RuntimeError(f"Hiçbir geçerli DICOM kesiti bulunamadı: {dicom_dir}")

        # En çok kesite sahip seriyi seç
        from collections import Counter
        series_counts = Counter(c[0] for c in all_candidates)
        best_sid = series_counts.most_common(1)[0][0]
        selected = [(fp, ds) for sid, fp, ds in all_candidates if sid == best_sid]

        if len(selected) < 3:
            raise RuntimeError(f"Yetersiz kesit sayısı ({len(selected)}): {dicom_dir}")

        # Z-pozisyonu yoksa InstanceNumber ile sırala
        def sort_key(item):
            ds = item[1]
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            return float(getattr(ds, "InstanceNumber", 0))

        selected.sort(key=sort_key)

        # Piksel verilerini yükle
        slices = []
        for fp, _ in selected:
            try:
                ds = pydicom.dcmread(fp, force=True)
                slices.append(ds)
            except Exception:
                pass

        if len(slices) < 3:
            raise RuntimeError(f"Yeterli kesit okunamadı: {len(slices)}")

        images = []
        for s in slices:
            arr = s.pixel_array.astype(np.float32)
            slope = float(getattr(s, "RescaleSlope", 1.0))
            intercept = float(getattr(s, "RescaleIntercept", 0.0))
            images.append(arr * slope + intercept)

        volume = np.stack(images, axis=-1)
        volume = np.swapaxes(volume, 0, 1)

        ps = slices[0].PixelSpacing if hasattr(slices[0], "PixelSpacing") else [1.0, 1.0]
        if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[1], "ImagePositionPatient"):
            dz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
            dz = dz if dz > 0 else float(getattr(slices[0], "SliceThickness", 1.0))
        else:
            dz = float(getattr(slices[0], "SliceThickness", 1.0))

        affine = np.diag([float(ps[0]), float(ps[1]), float(dz), 1.0])
        nib.save(nib.Nifti1Image(volume.astype(np.int16), affine), output_path)
        print(f"[SUCCESS] DICOM -> NIfTI tamamlandı (pydicom: {len(slices)} kesit, boyut: {volume.shape}).")
        return output_path

    except Exception as e:
        print(f"[ERROR] Tüm dönüştürücüler başarısız: {str(e)}")
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
