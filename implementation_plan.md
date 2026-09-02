# HepaRECIST-AI: 4D Longitudinal Abdominal Tümör Takibi ve Otomatik RECIST 1.1 Tedavi Yanıtı Değerlendirme Platformu

> **Durum:** Aktif Geliştirme & Model Eğitimi Aşaması  
> **Kapsam:** 55 Hasta / 106 BT Serisi (Sectra PACS DICOM) → Çok Zamanlı Abdominal Takip & RECIST 1.1 Kararı  
> **Hedef Bölge:** Tüm Batın (Abdominal Cavity: Karaciğer, Dalak, Böbrekler, Pankreas, Mide, Aorta, Lezyonlar)

---

## 🏗️ 1. Sistem Mimarisi ve 5 Fazlı Boru Hattı

```
[DICOM Serileri (t0, t1)] 
         │
         ▼
[Faz 1: Ön İşleme (preprocess.py)]
    • 106 Serilik PACS DICOM taraması (400-850 kesitlik aksiyel hacimler)
    • Çok katmanlı DICOM → 3D NIfTI dönüşümü (dicom2nifti + pydicom fallback)
    • Soft-Tissue HU Windowing (WL:40, WW:150) ve 1.0 mm³ izotropik B-Spline Resampling
         │
         ▼
[Faz 2: 3D Multi-Organ Segmentasyon (prepare_training_data.py & inference.py)]
    • TotalSegmentator ile otomatik 8-sınıflı batın maskeleme (Pseudo-Labeling):
      0: Arka Plan, 1: Karaciğer, 2: Dalak, 3: Böbrekler, 4: Pankreas, 5: Safra Kesesi, 6: Mide, 7: Aorta, 8: Lezyonlar
    • Dataset001_AbdominalTumor olarak nnU-Net v2 formatında veri seti hazırlığı
    • 3D Full-Resolution nnU-Net v2 derin öğrenme eğitimi (Google Colab / RunPod GPU)
         │
         ▼
[Faz 3: Radyomik Doğrulama (radiomics_module.py)]
    • HU yoğunluk istatistikleri ve 3D Küresellik (Sphericity Ψ) analizi
    • Basit kistlerin (0-20 HU, yüksek küresellik) ve damarların (≥120 HU) elenmesi
    • Güven Skoru üretimi (<0.75 olanlar için hekim onay bayrağı)
         │
         ▼
[Faz 4: 4D Longitudinal Registration & Eşleştirme (matching_engine.py)]
    • İki aşamalı SimpleITK hizalama: Rigid (Euler3D) + Deformable B-Spline Registration
    • Bipartite hibrit maliyet matrisi üzerinden Macar (Hungarian) Algoritması ile lezyon eşleştirme
    • Bireysel lezyon çap değişimleri (Δ%) ve Yeni Lezyon (New Lesion) tespiti
         │
         ▼
[Faz 5: Deterministik RECIST 1.1 Karar Motoru & Raporlama (main.py & YazOkuluDetectUI)]
    • Çapların Toplamı (SOD) ve Nadir takibi
    • Matematiksel kurallarla CR / PR / SD / PD sınıflandırması
    • C# .NET 9.0 WPF Masaüstü Arayüzü & Düzce Üniversitesi Formatında Otomatik PDF/JSON Klinik Rapor
```

---

## 📊 2. Veri Seti ve Etiketleme Yapısı

* **Kaynak:** Düzce Üniversitesi Tıp Fakültesi Araştırma Hastanesi Onkoloji Polikliniği (55 Hasta, 106 CT Serisi)
* **Primer Tanılar:** Kolon CA, Mide CA, Meme CA, Rektum CA, Pankreas CA, Akciğer CA
* **nnU-Net v2 Etiket Şeması (`dataset.json`):**

| Etiket No | Anatomik Doku / Yapı | Açıklama |
|:---:|:---|:---|
| **0** | `background` | Arka plan, kemik, hava ve diğer dokular |
| **1** | `liver` | Karaciğer parankim dokusu |
| **2** | `spleen` | Dalak dokusu |
| **3** | `kidneys` | Sağ ve sol böbrekler |
| **4** | `pancreas` | Pankreatik organ dokusu |
| **5** | `gallbladder` | Safra kesesi |
| **6** | `stomach` | Mide lümen/duvarı |
| **7** | `aorta` | Abdominal ana aort damarı |
| **8** | `lesion` | Solid tümör ve metastaz odakları |

---

## 🛠️ 3. Modül Haritası ve Dosya Sorumlulukları

| Modül Dosyası | Temel Sorumluluk | Çıktı / Etki |
|:---|:---|:---|
| `preprocess.py` | DICOM → 3D NIfTI, HU Windowing, 1mm³ B-Spline Resampling | `baseline.nii.gz`, `raw_hu_baseline.nii.gz` |
| `prepare_training_data.py` | 106 seriyi tarayıp TotalSegmentator ile 8-sınıflı batın maskeleri üretir | `01_nifti/`, `02_ts_masks/`, `03_merged_labels/` |
| `prepare_nnunet_data.py` | nnU-Net v2 dizin yapısını ve `dataset.json` dosyasını oluşturur | `nnUNet_raw/Dataset001_AbdominalTumor/` |
| `inference.py` | nnU-Net v2 model çıkarımı yapar (Model yoksa güvenli Mock modu) | 3D lezyon maskeleri ve Feret çapları |
| `radiomics_module.py` | Doku yoğunluğu, küresellik ve güven skoru hesaplar | Malign/Benign ayrımı ve `needs_review` bayrağı |
| `matching_engine.py` | B-Spline Registration + Hungarian Algorithm ile lezyon takibi | Eşleşen lezyonlar, Δ% çaplar, otomatik SOD |
| `report_generator.py` | Düzce Üniversitesi resmi hastane şablonunda radyoloji raporu üretir | Türkçe + İngilizce Standart Rapor (.txt / .pdf) |
| `pipeline.py` | Tüm 5 adımı tek komutla orkestre eden ana boru hattı | `run_full_pipeline()` sonuç dict'i |
| `main.py` | FastAPI asenkron REST API sunucusu (10 endpoint) | C# WPF masaüstü uygulamasıyla haberleşme |
| `YazOkuluDetectUI/` | C# .NET 9.0 WPF Masaüstü İstemcisi | Çift 3D kesit görüntüleyici, DICOM meta, RECIST rozetleri |

---

## 🚀 4. Uygulama ve Çalıştırma Adımları

1. **Google Colab'da Otomatik Veri Hazırlama:**
   ```python
   %cd /content/YazOkuluDetect
   !git pull origin main
   !python prepare_training_data.py --dicom_root "/content/drive/MyDrive/hasta dosya" --output_dir "/content/drive/MyDrive/Egitim_Verisi_Hazir"
   ```
2. **nnU-Net v2 Model Eğitimi:**
   ```bash
   nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
   nnUNetv2_train 001 3d_fullres 0
   ```
3. **Lokal Sunucu ve Masaüstü GUI Çalıştırma:**
   * Backend: `uvicorn main:app --reload`
   * Frontend: `dotnet run --project YazOkuluDetectUI\YazOkuluDetectUI.csproj`
