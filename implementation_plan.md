# Yapay Zeka Destekli Onkolojik BT Analizi — Uçtan Uca RECIST 1.1 Karar Destek Sistemi

> **Son Güncelleme:** 03 Ağustos 2026  
> **Kapsam:** Etiketlenmemiş Sectra PACS DICOM çiftleri → Longitudinal RECIST 1.1 Kararı  
> **Hedef Organ:** Üst Batın (Abdomen-Üst), Kontrastlı BT, 0.625 mm ince kesit

--- 

## Genel Bakış

İki aşamalı AI pipeline kullanan karar destek sistemi. nnU-Net tabanlı yüksek hassasiyetli segmentasyon; Radiomics/CNN tabanlı sınıflandırma; radyolog onay katmanı ve tamamen kural tabanlı RECIST 1.1 motoru. Elinizde **etiketlenmemiş Sectra PACS çıktısı (DICOM)** ve her hasta için **iki farklı tarihli çekim (Bazal + Kontrol)** bulunmaktadır.

---

## Veri Seti Durumu

| Özellik | Değer |
|---|---|
| **Format** | Sectra PACS CD/USB Export (DICOM) |
| **Yapı** | COMMON / DICOM / RA32 / RA64 / REPORTS / SECTRA klasörleri |
| **Kesit Kalınlığı** | 0.625 mm (THX) — Yüksek Z-çözünürlük |
| **Çekim Tipi** | Abdomen Üst — Kontrastlı BT |
| **Tarihler** | Her hasta için 2 çekim (Bazal + Kontrol) |
| **Etiket Durumu** | ❌ Etiketlenmemiş — Pseudo-labeling stratejisi uygulanacak |

---

## Danışman Onaylı Pipeline

```
CT Görüntüsü (DICOM — Sectra Export)
        │
        ▼
[1] DICOM → NIfTI Dönüştürücü  ← Python otomasyonu
    ├─ Sectra klasör yapısı tarama (DICOMDIR indeksi)
    ├─ Aksiyal + ince kesit serisi otomatik seçimi (≤1mm, >100 kesit)
    ├─ patient_01_baseline.nii.gz / patient_01_followup.nii.gz
    └─ HU Windowing (WL:50 WW:350 — Soft Tissue)
        │
        ▼
[2] Pseudo-Labeling (Yarı Otomatik Etiketleme)  ← SÜREÇ HIZLANDIRICI
    ├─ TotalSegmentator ile organ sınırları (Karaciğer, Dalak, Böbrek)
    ├─ LiTS/MSD Task03 pre-trained nnU-Net ile lezyon taslak maskesi
    └─ 3D Slicer'da radyolog hızlı onay/düzeltme (sıfırdan çizim YOK)
        │
        ▼
[3] 3D nnU-Net Segmentasyon  ← Yüksek sensitivity, düşük eşik
    └─ TÜM şüpheli bölgeleri tespit eder (kaçırma riskini minimize et)
        │
        ▼
[4] Radiomics / CNN Sınıflandırma
    ├─ Malign / Benign / Vasküler sınıflandırma
    ├─ PyRadiomics: texture, shape, HU histogram özellikleri
    ├─ Güven skoru (confidence) üretimi
    └─ Düşük güvenli bölgeler → Radyolog onay kuyruğu
        │
        ▼
[5] Radyolog Onay Adımı  ← Klinik güvenlik katmanı
    ├─ AI önerilerini güven skoru ile listeler
    ├─ Radyolog: onayla / reddet / düzenle
    └─ Onaylanan lezyonlar RECIST pipeline'a girer
        │
        ▼
[6] ANTsPy SyN Registration
    ├─ Bazal BT ↔ Kontrol BT Deformable (non-rigid) hizalama
    └─ Asenkron işlem (Celery + Redis) — UI bloklanmaz
        │
        ▼
[7] Hibrit Lezyon Eşleştirme  ← Projenin Özgün Katkısı
    ├─ Centroid 3D Öklid Mesafesi + IoU + Boyut Benzerliği
    ├─ Hungarian Algorithm (optimal global eşleştirme)
    └─ Yeni / Kaybolan lezyon tespiti
        │
        ▼
[8] RECIST 1.1 Uyumlu Ölçüm
    ├─ Hedef Lezyon Seçimi (maks 5 toplam, organ başına maks 2)
    ├─ 2D Feret Çapı (aksiyal dilim bazlı maksimum — 3D diyagonal DEĞİL)
    ├─ Lenf nodu: short-axis (≥15mm) | Organ: long-axis (≥10mm)
    └─ SOD (Sum of Diameters) hesabı
        │
        ▼
[9] Rule Engine — Açıklanabilir Karar  ← Tamamen kural tabanlı, LLM'siz
    └─ CR / PR / SD / PD kararı
        │
        ▼
[10] LLM Destekli Klinik Rapor  ← Karar değil, dil çevirisi
     └─ Doğrulanmış sayısal veriler → Türkçe + İngilizce radyoloji raporu
```

---

## Dört Bilimsel Katkı (Tez Eksenleri)

| # | Katkı | Açıklama | Düzey |
|---|-------|----------|-------|
| 1 | **Otomatik 3D Lezyon Segmentasyonu** | nnU-Net, pseudo-label destekli eğitim, yüksek sensitivity | Teknik uygulama |
| 2 | **Komşu Patoloji Ayrımı** ⭐ | Radiomics + CNN ile malign/benign/vasküler sınıflandırma + güven skoru | **Yeni katkı** |
| 3 | **Longitudinal Lezyon Eşleştirme** ⭐⭐ | Hibrit matching (centroid + IoU + boyut) + yeni/kaybolan lezyon tespiti | **En özgün katkı** |
| 4 | **Açıklanabilir Karar + Otomatik Raporlama** | Rule-based motor + LLM raporlayıcı | Klinik uygulama |

> [!IMPORTANT]
> **Katkı #3** (Longitudinal Eşleştirme) projenin bilimsel özgünlüğünü taşır.  
> **Katkı #2** hocanın "komşu patolojiden ayırt edebilir mi?" sorusuna doğrudan yanıt verir.  
> Bu ikili kombinasyon çalışmayı literatürdeki salt segmentasyon çalışmalarından belirgin biçimde ayırır.

---

## Teknik Mimari

```
┌─────────────────────────────────────────────────────────┐
│                  C# WPF / MAUI Arayüzü                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ DICOM Viewer │  │ Radyolog     │  │ RECIST        │  │
│  │ (FellowOak)  │  │ Onay Paneli  │  │ Rapor Görünüm │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────┬───────────────────────────┘
                              │ HTTP/REST (async polling)
┌─────────────────────────────▼───────────────────────────┐
│                   FastAPI Python Backend                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Celery + Redis  (Asenkron Görev Kuyruğu)        │   │
│  │  ├─ ANTsPy SyN Registration (uzun süre)          │   │
│  │  └─ nnU-Net Inference (GPU)                      │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │  DICOM→NIfTI Parser │  │  PyRadiomics             │  │
│  │  Seri Seçici        │  │  ML Sınıflandırıcı       │  │
│  └─────────────────────┘  └──────────────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  RECIST 1.1      │  │    LLM Rapor Modülü          │  │
│  │  Rule Engine     │  │  (sadece dil dönüşümü)       │  │
│  │  CR/PR/SD/PD     │  │  Gemini / GPT-4o / Lokal LLM│  │
│  └──────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Kritik Teknik Detaylar

### DICOM Seri Seçici
```python
def select_target_series(study_dir: str) -> str:
    """
    Sectra DICOMDIR üzerinden aksiyal + ince kesit serisi otomatik seçimi.
    """
    for series in read_dicomdir(study_dir):
        if (series.slice_thickness <= 1.0
                and "AXIAL" in series.image_type
                and series.slice_count > 100):
            return series
```

### 2D RECIST Çap Ölçümü (Aksiyal Dilim Bazlı)
```python
def measure_recist_diameter(mask_3d: np.ndarray) -> float:
    """
    3D maskeyi aksiyal (Z) dilim dilim tara,
    her dilimdeki max 2D Feret çapını bul.
    RECIST 1.1: 3D diyagonal çap KULLANILMAZ.
    """
    max_diameter = 0.0
    for z in range(mask_3d.shape[2]):
        slice_2d = mask_3d[:, :, z]
        if slice_2d.any():
            diameter = feret_diameter_max(slice_2d)
            max_diameter = max(max_diameter, diameter)
    return max_diameter
```

### Hedef Lezyon Seçici (RECIST 1.1 Kısıtları)
```python
def select_target_lesions(lesions: list) -> list:
    """
    RECIST 1.1: Toplam maks 5 lezyon, organ başına maks 2.
    Organ lezyonu: long-axis ≥10mm
    Lenf nodu: short-axis ≥15mm
    """
    organ_counts = defaultdict(int)
    targets = []
    for lesion in sorted(lesions, key=lambda l: l.diameter, reverse=True):
        if organ_counts[lesion.organ] < 2 and len(targets) < 5:
            if lesion.is_lymph_node and lesion.short_axis >= 15:
                targets.append(lesion)
                organ_counts[lesion.organ] += 1
            elif not lesion.is_lymph_node and lesion.long_axis >= 10:
                targets.append(lesion)
                organ_counts[lesion.organ] += 1
    return targets
```

### RECIST 1.1 Karar Motoru
```python
def recist_decision_engine(sod_baseline: float,
                            sod_followup: float,
                            new_lesion: bool) -> str:
    """
    Tamamen deterministik, LLM bağımsız karar motoru.
    Kaynak: Eisenhauer et al., 2009, Eur J Cancer
    """
    change_pct = (sod_followup - sod_baseline) / sod_baseline * 100

    if new_lesion or (change_pct >= 20 and (sod_followup - sod_baseline) >= 5):
        return "PD"   # Progressive Disease
    if sod_followup == 0:
        return "CR"   # Complete Response
    if change_pct <= -30:
        return "PR"   # Partial Response
    return "SD"       # Stable Disease
```

### LLM Rapor Modülü (Doğru Kullanım)
```python
# ✅ DOĞRU: LLM'e sadece doğrulanmış sayılar ve kesinleşmiş karar gönderilir
prompt = f"""
Sen bir radyoloji rapor yazarısın.
Aşağıdaki DOĞRULANMIŞ klinik verileri standart radyoloji raporu diline çevir.
Karar değiştirme, yorum ekleme.

Bazal SOD  : {sod_baseline:.1f} mm
Kontrol SOD: {sod_followup:.1f} mm
Değişim    : {change_pct:.1f}%
Yeni Lezyon: {"Var" if new_lesion else "Yok"}
RECIST Kararı: {recist_decision}   ← Bu değiştirilemez

Raporu Türkçe ve İngilizce olarak yaz.
"""

# ❌ YANLIŞ: LLM'e ham görüntü gönderip karar isteme
```

---

## Geliştirme Fazları

### Faz 1 — Altyapı & Veri Hazırlığı (4–6 hafta)
- [ ] Sectra DICOM klasör tarayıcı + seri seçici scripti
- [ ] DICOM → NIfTI dönüştürücü (`dicom2nifti` + HU windowing)
- [ ] `patient_XX_baseline.nii.gz` / `patient_XX_followup.nii.gz` klasör yapısı
- [ ] TotalSegmentator ile organ sınırı pseudo-labeling
- [ ] LiTS / MSD Task03 pre-trained nnU-Net ile lezyon taslak maskesi üretimi
- [ ] 3D Slicer'da radyolog hızlı maske onayı (sıfırdan çizim YOK)
- [ ] FastAPI iskelet projesi kurulumu
- [ ] Celery + Redis asenkron görev altyapısı
- [ ] C# WPF prototip (DICOM viewer + radyolog onay paneli)

### Faz 2 — Segmentasyon Modeli Eğitimi (6–8 hafta)
- [ ] nnU-Net veri hazırlığı (`imagesTr/labelsTr` klasör yapısı)
- [ ] RunPod / Vast.ai ortam kurulumu (RTX 4090)
- [ ] nnU-Net fingerprint + plan oluşturma (düşük eşik konfigürasyonu)
- [ ] 5-fold cross-validation eğitimi
- [ ] Dice skoru değerlendirme (hedef: >0.85)
- [ ] Model weights lokal inference için dışa aktarım

### Faz 3 — Radiomics Sınıflandırma Modülü (3–4 hafta) ⭐
- [ ] PyRadiomics entegrasyonu (texture, shape, HU histogram özellikleri)
- [ ] Malign / Benign / Vasküler etiketli eğitim seti hazırlama
- [ ] Random Forest / XGBoost sınıflandırıcı eğitimi
- [ ] Güven skoru (confidence) sistemi
- [ ] Düşük güvenli bölge → radyolog onay kuyruğu mekanizması
- [ ] **Ablation study:** Sadece nnU-Net vs nnU-Net + Radiomics (FP oranı karşılaştırması)

### Faz 4 — Longitudinal Analiz Motoru (4–6 hafta) ⭐⭐
- [ ] ANTsPy SyN deformable registration (Celery async olarak)
- [ ] Hibrit lezyon eşleştirme algoritması (centroid + IoU + boyut)
- [ ] Hungarian algorithm entegrasyonu
- [ ] Yeni / kaybolan lezyon tespiti
- [ ] 2D Feret çapı (aksiyal dilim bazlı) + SOD hesaplama
- [ ] RECIST 1.1 hedef lezyon seçici (maks 5, organ başına maks 2)

### Faz 5 — Karar Motoru & Raporlama (3–4 hafta)
- [ ] RECIST 1.1 rule engine (tamamen kural tabanlı, LLM'siz)
- [ ] LLM rapor modülü (Gemini / GPT-4o / Lokal LLM entegrasyonu)
- [ ] PDF rapor üretimi (Türkçe + İngilizce)
- [ ] C# UI → FastAPI tam entegrasyon (async polling + radyolog onay akışı)

### Faz 6 — Doğrulama & Yayın Hazırlığı (4 hafta)
- [ ] Radyolog pilot değerlendirmesi
- [ ] İstatistiksel analiz (ICC, kappa katsayısı, sensitivity/specificity)
- [ ] Ablation study sonuçlarının raporlanması
- [ ] Makale taslağı hazırlama

---

## Açık Kararlar (Yanıtlanması Gereken)

> [!WARNING]
> Aşağıdaki kararlar teknik implementasyonu doğrudan etkiler.

| # | Soru | Seçenekler |
|---|------|-----------|
| 1 | **LLM tercihi?** | Gemini API (ücretsiz tier) / GPT-4o (ücretli) / Lokal LLaMA 3 (gizlilik) |
| 2 | **UI framework?** | WPF (.NET — olgun) / MAUI (.NET 8 — cross-platform) / ASP.NET Core (web) |
| 3 | **Eğitim ortamı?** | RunPod RTX 4090 (kiralık) / Lokal GPU (varsa) / Google Colab Pro |

---

## Yayın Potansiyeli

| Dergi | Seviye | Uygun Katkı |
|-------|--------|-------------|
| Medical Image Analysis | Q1 | Hibrit lezyon eşleştirme |
| European Journal of Radiology | Q2 | RECIST otomasyon sistemi |
| Computers in Biology and Medicine | Q2 | Ablation study + radyomik sınıflandırma |

> [!TIP]
> **Ablation study** (hibrit matching vs. naive centroid-only vs. IoU-only karşılaştırması) yayın özgünlüğünü güçlü şekilde kanıtlar ve tek başına makaleye değer bir bulgu üretir.

---

## ✅ Eğitim Sonrası Yapılacaklar Listesi

> [!IMPORTANT]
> Model eğitimi tamamlanıp `./models/checkpoint_final.pth` dosyası yerleştirildikten sonra aşağıdaki adımlar **sırayla** uygulanmalıdır.

### 1. Model Ağırlığını Yerleştir
```
c:\Projects\YazOkuluDetect\models\checkpoint_final.pth
```
Başka hiçbir kod değişikliği gerekmez — sistem otomatik olarak gerçek AI moduna geçer.
Arayüzdeki rozet: **"🔧 Test Modu"** → **"🧠 Gerçek AI Modu (nnU-Net)"** olarak güncellenir.

---

### 2. B-Spline Registration Parametrelerini Klinik Moda Al

> [!WARNING]
> **Bu adım kritiktir.** Şu an B-Spline registration parametreleri test/geliştirme ortamı için hızlandırılmış değerlere ayarlıdır.
> Klinik kullanım öncesinde [`matching_engine.py`](file:///c:/Projects/YazOkuluDetect/matching_engine.py) dosyasında şu değişikliği yapın:

**Dosya:** [`matching_engine.py`](file:///c:/Projects/YazOkuluDetect/matching_engine.py) — yaklaşık satır 77-104

```python
# ❌ MEVCUT (Test/Hızlı mod — eğitim sonrası bunu DEĞİŞTİRİN):
grid_physical_spacing = [80.0, 80.0, 80.0]   # kaba grid
numberOfIterations    = 50                     # az iterasyon
maximumNumberOfFunctionEvaluations = 300       # az değerlendirme
shrinkFactors = [2]                            # tek piramit seviyesi

# ✅ KLİNİK MOD (eğitim sonrası buna GEÇİN):
grid_physical_spacing = [50.0, 50.0, 50.0]   # hassas grid
numberOfIterations    = 100                    # yeterli iterasyon
maximumNumberOfFunctionEvaluations = 1000      # tam değerlendirme
shrinkFactors = [2, 1]                         # iki piramit seviyesi
```

**Neden önemli:** 50mm grid spacing, organ kayması düzeltmede ~2-3mm daha hassas registration sağlar. Hatalı registration → yanlış lezyon eşleştirme → yanlış SOD → yanlış RECIST kararı zincirini önler.

---

### 3. Sistem Doğrulama Testleri
- [ ] En az 5 hasta çiftiyle (t0 + t1) end-to-end pipeline testi
- [ ] RECIST kararlarını radyolog referans kararlarıyla karşılaştır
- [ ] Registration kalitesini görsel olarak doğrula (hizalanmış görüntüleri kontrol et)
- [ ] Lezyon eşleştirme oranını hesapla (yanlış eşleşme var mı?)

---

### 4. Opsiyonel: LLM Rapor Modunu Aktif Et
`main.py` — `/pipeline` endpoint çağrısında:
```json
{ "use_llm": true, "llm_api_key": "YOUR_GEMINI_API_KEY" }
```
LLM **karar vermez** — sadece doğrulanmış RECIST metriklerini klinik rapor diline çevirir.

