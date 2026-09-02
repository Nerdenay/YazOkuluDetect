# HepaRECIST-AI: 4D Longitudinal Abdominal Tümör Takibi ve Otomatik RECIST 1.1 Tedavi Yanıtı Değerlendirme Platformu

## 📖 BÖLÜM 1: Sistem Mimarisi ve Faz Bazlı Fonksiyonel Rapor

Sistemimiz, medikal BT taramalarından 3D organ ve tümör/metastaz tespiti yapıp RECIST 1.1 standartlarına uygun deterministik tedavi yanıtı analizi üreten ve bunu akıcı raporlara dönüştüren **5 Fazlı Tam Katmanlı Yazılım Mimarisine** sahiptir.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 C# WPF MASAÜSTÜ ARAYÜZÜ                 │
                  │   (fo-dicom Metadata, Karar Rozetleri, Rapor Paneli)    │
                  └────────────────────────────┬────────────────────────────┘
                                               │ HTTP / JSON API
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │              FastAPI BACKEND SUNUCUSU (main.py)         │
                  └────────────────────────────┬────────────────────────────┘
                                               │
  ┌───────────────────┬───────────────────┼───────────────────┬───────────────────┐
  ▼                   ▼                   ▼                   ▼                   ▼
[FAZ 1]             [FAZ 2]             [FAZ 3]             [FAZ 4]             [FAZ 5]
Preprocessing       AI Segmentasyon     Radiomics Classif.  Registration        Orchestrator
(preprocess.py)     (inference.py)      (radiomics_module)  (matching_engine)   (pipeline.py)
HU Windowing        nnU-Net 3D AI       Malign / Benign     Rigid + B-Spline    Report Generator
Resampling (1mm³)   / Safe Mock         / Vasküler          Hungarian Match     (Template/LLM)
```

---

### Faz Bazlı Detaylı Fonksiyon Raporu

#### 🛠️ Faz 1: Altyapı, Preprocessing & Masaüstü Arayüzü
* **`preprocess.py`**:
  * Ham DICOM serilerini `dicom2nifti` kütüphanesi ile 3D NIfTI (`.nii.gz`) dosyasına dönüştürür.
  * Soft tissue / karaciğer için **Hounsfield Unit (HU) Windowing** uygular ($\text{WL}=40, \text{WW}=150$, $[-35, +115]\text{ HU}$ aralığı).
  * Tüm boyutları `SimpleITK.ResampleImageFilter` ile $1.0\text{ mm} \times 1.0\text{ mm} \times 1.0\text{ mm}$ izotropik çözünürlüğe getirir.
* **`main.py` & `YazOkuluDetectUI`**:
  * Uvicorn tabanlı FastAPI sunucusu ve C# WPF modern dark-mode arayüzü. `fo-dicom` ile hasta ID, Ad ve Çekim Tarihi DICOM başlıklarından okunur.

#### 📦 Faz 2: nnU-Net v2 Veri Yapılandırması & Inference Boru Hattı
* **`prepare_nnunet_data.py`**:
  * Vaka sayısından bağımsız olarak (3 vaka veya 500+ vaka) klasördeki tüm NIfTI dosyalarını tarar ve nnU-Net v2 `Dataset001_LiverLesion` hiyerarşisine (`imagesTr`, `labelsTr`) dönüştürür.
  * Otomatik `dataset.json` dosyasını üretir ve RunPod GPU sunucuları için `run_cloud_training.sh` Bash script'ini hazırlar.
* **`inference.py`**:
  * Model klasöründe (`./models`) eğitilmiş ağırlıklar (`checkpoint_final.pth`) varsa **gerçek nnU-Net 3D AI çıkarımını** çalıştırır.
  * Ağırlıklar henüz yüklenmemişse geliştirme/test ortamını aksatmamak için **güvenli Mock Modu** ile simüle 3D maske üretir.

#### 🧠 Faz 3: Radiomics Sınıflandırma Modülü (Komşu Patoloji Ayrımı)
* **`radiomics_module.py`**:
  * 3D lezyon adayları için **Şekil (Yuvarlaklık/Sphericity)**, **Doku Heterojenliği (Entropi/GLCM)** ve **HU Yoğunluk İstatistikleri (Mean, Std, Percentiles)** çıkarır.
  * Adayları **Malign (Tümör/Metastaz)**, **Benign (Kist: 0-20 HU, homojen)** veya **Vasküler/Kalsifikasyon (>120 HU)** olarak sınıflandırır.
  * Her lezyon için $0.0 - 1.0$ arası Güven Skoru (Confidence Score) üretir. Skor $<0.75$ ise `needs_review = True` bayrağı ile C# arayüzündeki **Radyolog Onay Kuyruğu**'na sevk eder.

#### 🎯 Faz 4: Longitudinal Registration & Hibrit Lezyon Eşleştirme Motoru
* **`matching_engine.py`**:
  * **SimpleITK Registration**: Baseline ($t_0$) ve Follow-up ($t_1$) BT taramalarını Rigid (Euler3D) + B-Spline Deformable registration ile hizalar. Hasta nefes ve organ kaymaları düzeltilir.
  * **Hibrit Hungarian Algorithm**: $t_0$ ve $t_1$ lezyonlarını 3D Öklid mesafesi ($\%50$), Hacim benzerliği ($\%30$) ve Çap oranı ($\%20$) bileşik maliyet matrisi üzerinden `scipy.optimize.linear_sum_assignment` ile optimal şekilde eşleştirir.
  * Eşleşmeyen lezyonlardan **Yeni Lezyon** veya **Kaybolan Lezyon** tespiti yapar ve RECIST 1.1 SOD (Sum of Diameters) değerlerini otomatik hesaplar.

#### 📝 Faz 5: Uçtan Uca Entegrasyon & Klinik Rapor Formatlayıcı
* **`report_generator.py`**:
  * **Şablon Modu (Offline)**: İnternet veya API anahtarı gerektirmeden standart onkolojik radyoloji raporu üretir.
  * **LLM Modu (Gemini API)**: Doğrulanmış metrikleri akıcı klinik dile çevirir. **LLM ASLA karar vermez**, sadece formatlama yapar.
* **`pipeline.py`**:
  * `run_full_pipeline()`: Preprocessing → Segmentasyon → Radiomics → Registration → Hungarian Eşleştirme → RECIST 1.1 Kararı → Rapor Üretimi adımlarını tek tıkla çalıştırır.

---

## 🚀 BÖLÜM 2: Model Eğitimi ve Yayına Alma "How-To" Rehberi

Yazılım mimarimiz tam olduğu için, model eğitimi öncesinde ve sonrasında yapılması gereken adım adım işlemler şunlardır:

### 📍 Adım 1: Elinizdeki Verileri İşleme (Lokal Veri Seti Hazırlığı)

1. Bilgisayarınızda ham DICOM veya NIfTI dosyalarınızı şu dizin yapısında düzenleyin:
   ```
   c:\Projects\YazOkuluDetect\my_raw_data\
     ├── images\   (BT NIfTI dosyaları: case_001.nii.gz, case_002.nii.gz ...)
     └── labels\   (Maske NIfTI dosyaları: case_001.nii.gz, case_002.nii.gz ...)
   ```
2. Python sanal ortamınızda veri setini nnU-Net formatına çevirin:
   ```powershell
   cd c:\Projects\YazOkuluDetect
   .venv\Scripts\python -c "from prepare_nnunet_data import setup_nnunet_environment, auto_scan_and_convert, generate_cloud_training_script; paths = setup_nnunet_environment('./nnunet_data', dataset_id=1, dataset_name='LiverLesion'); auto_scan_and_convert('./my_raw_data/images', './my_raw_data/labels', paths); generate_cloud_training_script(paths)"
   ```
3. Bu komut `./nnunet_data/nnUNet_raw/Dataset001_LiverLesion` klasörünü, `dataset.json` dosyasını ve `run_cloud_training.sh` betiğini otomatik üretecektir.

---

### 📍 Adım 2: Lokal Sistem Testi (Smoke Test)

Eğitime geçmeden önce backend ve arayüzün çalıştığını doğrulayın:

1. **FastAPI Sunucusunu Çalıştırın:**
   ```powershell
   cd c:\Projects\YazOkuluDetect
   .venv\Scripts\uvicorn main:app --reload
   ```
   * Tarayıcıdan Swagger dokümantasyonunu kontrol edin: `http://127.0.0.1:8000/docs`

2. **C# WPF Arayüzünü Çalıştırın:**
   ```powershell
   dotnet run --project YazOkuluDetectUI\YazOkuluDetectUI.csproj
   ```
   * DICOM klasörünü seçin, "Ön İşleme", "AI Segmentasyon (Mock)" ve "RECIST Kararı Al" butonlarına basarak akışın tamamlandığını görün.

---

### 📍 Adım 3: Bulut (RunPod / Vast.ai) Üzerinde Model Eğitimi

1. **RunPod GPU Kiralama:**
   * RunPod.io veya Vast.ai üzerinden 1x **NVIDIA RTX 4090 (24GB VRAM)** veya **A100 GPU** pod'u başlatın (PyTorch 2.x şablonu ile).
2. **nnU-Net v2 Kurulumu (Pod İçinde Terminalde):**
   ```bash
   pip install nnunetv2
   ```
3. **Verileri Yükleme:**
   * Bilgisayarınızda hazırlanan `./nnunet_data` klasörünü ve `run_cloud_training.sh` dosyasını `scp` veya RunPod Web UI üzerinden sunucuya yükleyin.
4. **Eğitimi Başlatma:**
   ```bash
   chmod +x run_cloud_training.sh
   ./run_cloud_training.sh
   ```
   * Betik otomatik olarak `nnUNetv2_plan_and_preprocess` çalıştıracak, ardından 5-fold cross validation eğitimini (Fold 0) başlatacaktır.

---

### 📍 Adım 4: Eğitilmiş Ağırlıkları Sisteme Entegre Etme (Tak-Çalıştır)

1. Eğitim tamamlandığında RunPod üzerindeki `nnUNet_results/Dataset001_LiverLesion/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/` klasöründen şu iki dosyayı bilgisayarınıza indirin:
   * `checkpoint_final.pth`
   * `dataset.json`

2. İndirdiğiniz dosyaları projenizin `./models` klasörüne yapıştırın:
   ```
   c:\Projects\YazOkuluDetect\models\
     ├── checkpoint_final.pth
     └── dataset.json
   ```

3. **Canlı AI Kontrolü:**
   * FastAPI sunucusu artık `./models` klasöründe `.pth` dosyasını otomatik tespit edecek, **Mock modunu kapatacak** ve gelen BT görüntülerine **gerçek 3D nnU-Net AI çıkarımı** uygulayacaktır. Sistemde başka hiçbir kod değişikliği yapılması gerekmez!

---

### 📝 Özet Tablo: Modül ve Çıktı Rehberi

| Modül Dosyası | Temel İşlevi | Çıktı / Etki |
|---------------|--------------|--------------|
| `preprocess.py` | DICOM → NIfTI, HU Windowing, Resampling | `baseline.nii.gz`, `raw_hu_baseline.nii.gz` |
| `prepare_nnunet_data.py` | Scalable dataset converter | `Dataset001_AbdominalTumor/`, `dataset.json`, `run_cloud_training.sh` |
| `prepare_training_data.py` | Otomatik Batın Veri Hazırlama & Pseudo-Labeling | `01_nifti/`, `02_ts_masks/`, `03_merged_labels/` |
| `inference.py` | nnU-Net v2 / Safe Mock inference | `baseline_mask.nii.gz`, 3D Hacim & Çap |
| `radiomics_module.py` | Şekil, doku, HU sınıflandırma | Malign/Benign/Vasküler + Confidence + `needs_review` |
| `matching_engine.py` | Registration + Hungarian Algorithm | Rigid+B-Spline NIfTI, Eşleşen Lezyonlar, Otomatik SOD |
| `report_generator.py` | Klinik Rapor üretici | Türkçe + İngilizce Standart Radyoloji Raporu (.txt) |
| `pipeline.py` | Uçtan uca orkestrasyon | Tüm pipeline'ı çalıştıran `run_full_pipeline()` |
| `main.py` | FastAPI REST API (10 endpoint) | C# WPF haberleşmesi (`http://127.0.0.1:8000`) |
| `YazOkuluDetectUI` | C# WPF Masaüstü Uygulaması | Kullanıcı arayüzü, DICOM viewer, Onay paneli |
