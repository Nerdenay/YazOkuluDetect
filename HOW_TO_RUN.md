# HepaRECIST-AI: 4D Longitudinal Abdominal Tümör Takibi ve Otomatik RECIST 1.1 Tedavi Yanıtı Değerlendirme Platformu
## 🚀 Hızlı Başlangıç ve Çalıştırma Rehberi (How-To-Run)

Bu proje, BT (Bilgisayarlı Tomografi) taramalarından 3D organ ve tümör/metastaz tespiti, komşu patoloji ayrımı (Radiomics), çok zamanlı boylamsal lezyon takibi (B-Spline Registration + Hungarian Algorithm) ve RECIST 1.1 standartlarında deterministik tedavi yanıtı analizi sunan uçtan uca otomatik bir onkolojik analiz platformudur.

Proje 2 temel bileşenden oluşmaktadır:
1. **Python FastAPI Backend Sunucusu:** AI modellerini, görüntü işleme ve karar motorunu çalıştırır (`http://127.0.0.1:8000`).
2. **C# WPF Masaüstü Arayüzü (`YazOkuluDetectUI`):** Radyoloğun/kullanıcının DICOM yükleyip tüm analizi görsel olarak takip ettiği modern masaüstü uygulaması.

---

## 📋 Sistem Gereksinimleri

- **İşletim Sistemi:** Windows 10 / 11
- **Python:** Python 3.10 veya üzeri
- **.NET SDK:** .NET 9.0 SDK (C# WPF masaüstü uygulamasını çalıştırmak için)
- *(İsteğe Bağlı)* **Visual Studio 2022** veya **VS Code**

---

## 🛠️ Adım Adım Kurulum ve Çalıştırma

### A) 1. ADIM: Python Backend Sunucusunu Çalıştırma

1. Proje ana klasöründe bir terminal (PowerShell veya Komut İstemi) açın:
   ```powershell
   cd c:\Projects\YazOkuluDetect
   ```

2. Python sanal ortamını (venv) oluşturun (eğer henüz oluşturulmadıysa):
   ```powershell
   python -m venv .venv
   ```

3. Sanal ortamı aktifleştirin:
   ```powershell
   .venv\Scripts\activate
   ```

4. Gerekli Python kütüphanelerini yükleyin:
   ```powershell
   pip install -r requirements.txt
   ```

5. FastAPI backend sunucusunu başlatın:
   ```powershell
   uvicorn main:app --reload
   ```

6. **Doğrulama:** Sunucu başladığında terminalde şu çıktıyı göreceksiniz:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   ```
   * Dilerseniz tarayıcınızdan `http://127.0.0.1:8000/docs` adresine giderek Swagger etkileşimli API test arayüzünü inceleyebilirsiniz.

---

### B) 2. ADIM: C# WPF Masaüstü Uygulamasını Çalıştırma

Python sunucusu arka planda çalışırken yeni bir terminal penceresi açın:

#### Yöntem 1: Terminal Üzerinden (En Hızlı Yöntem)
```powershell
cd c:\Projects\YazOkuluDetect
dotnet run --project YazOkuluDetectUI\YazOkuluDetectUI.csproj
```

#### Yöntem 2: Visual Studio ile
1. `YazOkuluDetectUI\YazOkuluDetectUI.csproj` dosyasını Visual Studio 2022 ile açın.
2. Yukarıdaki **Start (Başlat / F5)** butonuna basarak uygulamayı çalıştırın.

---

## 🔬 Uygulamanın Kullanımı ve Test Akışı

Uygulama açıldığında sol üstte **"Python API: Bağlı (http://127.0.0.1:8000)"** yeşil durum yazısını göreceksiniz.

1. **DICOM Klasörü Seçimi:**
   * **"DICOM Klasörü Seç"** butonuna tıklayın ve bilgisayarınızdaki örnek bir DICOM tetkik klasörünü seçin.
   * `fo-dicom` kütüphanesi otomatik olarak **Hasta ID**, **Hasta Adı** ve **Çekim Tarihi** bilgilerini ekrana yansıtacaktır.

2. **Görüntü Ön İşleme (Preprocessing):**
   * **"1. DICOM Ön İşleme"** butonuna tıklayın.
   * Sunucu DICOM serisini NIfTI (`.nii.gz`) formatına dönüştürecek, HU Windowing (WL:40, WW:150) uygulayacak ve $1.0\text{ mm}^3$ izotropik çözünürlüğe yeniden örnekleyecektir (Resampling).

3. **AI Segmentasyon (Lezyon Tespiti):**
   * **"2. AI Segmentasyon (Mock)"** butonuna tıklayın.
   * Sunucu 3D lezyon maskesini üretecek, lezyon hacmini ($mm^3$) ve 2D Feret çapını ($mm$) hesaplayacaktır.

4. **RECIST 1.1 Karar Motoru & Raporlama:**
   * **"3. RECIST 1.1 Kararı Al"** butonuna tıklayın.
   * Deterministik kural motoru (Eisenhauer et al., 2009) **CR (Complete Response)**, **PR (Partial Response)**, **SD (Stable Disease)** veya **PD (Progressive Disease)** kararını üretecek ve renkli klinik karar rozetini güncelleyecektir.
   * Sağ taraftaki raporda Türkçe yapılandırılmış klinik karar metni görüntülenecektir.

---

## 📁 Proje Modül Haritası (Akademik İnceleme İçin)

Kod yapısını incelemek isteyenler için ana modüller ve işlevleri:

| Dosya Adı | Sorumluluk / İşlev |
|-----------|--------------------|
| **`preprocess.py`** | DICOM → NIfTI dönüşümü, HU windowing, 1mm³ izotropik resampling. |
| **`inference.py`** | nnU-Net v2 AI çıkarım motoru. Model ağırlıkları yoksa güvenli Mock fallback modunda çalışır. |
| **`radiomics_module.py`** | 3D lezyon adayları için Şekil, Doku (Entropi) ve HU istatistikleri çıkararak **Malign / Benign / Vasküler** ayrımı yapar ve Güven Skoru üretir. |
| **`matching_engine.py`** | SimpleITK Rigid + B-Spline Deformable Registration ve **Hibrit Hungarian Algorithm** ile 2 zamanlı ($t_0 \leftrightarrow t_1$) lezyon takibi yapar. |
| **`report_generator.py`** | Deterministik sonuçları standart onkolojik radyoloji raporu formatına dönüştürür (Şablon veya Gemini LLM modu). |
| **`pipeline.py`** | Tüm 6 adımı orkestre eden ana iş akışı motoru (`run_full_pipeline`). |
| **`prepare_nnunet_data.py`** | Herhangi bir NIfTI veri setini nnU-Net v2 formatına dönüştüren ve RunPod GPU eğitim script'ini üreten araç. |
| **`main.py`** | Tüm modülleri dış dünyaya açan FastAPI REST API sunucusu. |
| **`YazOkuluDetectUI/`** | C# WPF masaüstü arayüz projesi (`MainWindow.xaml` ve `MainWindow.xaml.cs`). |

---

## ❓ Sıkça Sorulan Sorular / Sorun Giderme

* **Soru: Model eğitimi henüz yapılmadı mı?**
  * **Cevap:** Sistem mimarisi %100 tamamlanmıştır. Ağırlıklar yüklenene kadar sistem **güvenli Mock modunda** çalışarak tüm boru hattını eksiksiz test etmeye imkan verir. Eğitilmiş `.pth` model dosyası `./models/` klasörüne konduğu an sistem otomatik olarak canlı AI moduna geçer.
* **Soru: `dotnet` komutu bulunamadı hatası alıyorum.**
  * **Cevap:** Bilgisayarınızda .NET 9.0 SDK yüklü olduğundan emin olun veya projeyi doğrudan Visual Studio 2022 ile açıp F5'e basın.
