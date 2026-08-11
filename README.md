# Yapay Zeka Destekli Onkolojik BT Analizi — RECIST 1.1 Karar Destek Sistemi

Bu proje, medikal Bilgisayarlı Tomografi (BT) taramalarından 3D karaciğer lezyon tespiti, komşu patoloji sınıflandırması (Radiomics), 2 zamanlı lezyon takibi (Registration + Hungarian Algorithm) ve RECIST 1.1 standartlarında deterministik karar desteği sunan uçtan uca bir klinik karar destek sistemidir.

Detaylı çalıştırma ve test rehberi için lütfen **[HOW_TO_RUN.md](HOW_TO_RUN.md)** dosyasına bakınız.

---

## ⚡ Hızlı Başlangıç

### 1. Python Backend Sunucusu (FastAPI)
```powershell
# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Sunucuyu başlatın
uvicorn main:app --reload
```
* Tarayıcı API Dokümantasyonu: `http://127.0.0.1:8000/docs`

### 2. C# WPF Masaüstü Uygulaması (`YazOkuluDetectUI`)
```powershell
dotnet run --project YazOkuluDetectUI\YazOkuluDetectUI.csproj
```

---

## 🏗️ Proje Mimarisi

```
DICOM (t0 + t1) 
    │
    ▼
[1] Preprocessing (preprocess.py)  ── HU Windowing + 1mm³ Isotropic Resampling
    │
    ▼
[2] AI Segmentasyon (inference.py) ── 3D nnU-Net / Safe Mock Mode
    │
    ▼
[3] Radiomics (radiomics_module.py) ── Malign / Benign / Vasküler + Güven Skoru
    │
    ▼
[4] Registration & Match (matching_engine.py) ── Rigid + B-Spline + Hungarian Alg.
    │
    ▼
[5] RECIST 1.1 Karar Motoru (main.py) ── CR / PR / SD / PD (Deterministik)
    │
    ▼
[6] Raporlayıcı (report_generator.py) ── Türkçe + İngilizce Klinik Rapor
```
