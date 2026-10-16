# HepaRECIST-AI: Hekim İçin BT Lezyon Etiketleme Kılavuzu (ITK-SNAP)
### (Karaciğer + Akciğer Metastaz Takibi)

Sayın Hocam / Değerli Meslektaşımız,

Onkolojik RECIST 1.1 değerlendirmesinde en sık metastaz görülen iki ana bölge olan **Karaciğer (Abdomen)** ve **Akciğer (Toraks Parankimi)** serileri yapay zeka tarafından taranmış ve ilgili organlar otomatik olarak etiketlenmiştir.

Sizden beklenen işlem; 
1. **Abdomen Serisinde:** Karaciğer içerisindeki primer tümör / metastaz odaklarını,
2. **Toraks Serisinde:** Akciğer parankimi içerisindeki metastatik nodül/tümör odaklarını,
**Label 8 (Lezyon)** olarak fırça ile işaretlemenizdir.

---

## 🚀 4 Adımda Hızlı Etiketleme

### 1. Adım: Dosyaları ITK-SNAP'e Yükleme
1. **ITK-SNAP** programını açın.
2. **File ➔ Open Main Image** seçeneğine tıklayın.
   - İlgili hastanın BT dosyasını seçin (Örn: `hasta_001_t0_abd_0000.nii.gz` veya `hasta_001_t0_thorax_0000.nii.gz`).
3. **Segmentation ➔ Open Segmentation** seçeneğine tıklayın.
   - Aynı hastanın maske dosyasını seçin (Örn: `hasta_001_t0_abd.nii.gz` veya `hasta_001_t0_thorax.nii.gz`).
   - *(Organlar renkli katman olarak BT'nin üzerine oturacaktır).*

---

### 2. Adım: "Label 8 (Lezyon)" Etiketini Ekleme / Seçme
1. Sol taraftaki **Segmentation Labels** paneline bakın.
2. Listenin altındaki **"New"** (veya **"+"**) butonuna tıklayın.
3. Açılan küçük pencerede:
   - **Label ID:** `8` yazın.
   - **Label Name:** `Lesion` (veya `Metastasis`) yazın.
   - **Color (Renk):** Belirgin olması için **Kırmızı (Red)** seçin.
4. **Apply / OK** deyin ve **Active Label** olarak **Label 8: Lesion**'ı seçin.

---

### 3. Adım: Pencereleme (Window/Level) ve Boyama
1. **Pencere Seçimi:**
   - **Karaciğer İçin (Abdomen):** Yumuşak doku penceresi (**WL: 40, WW: 150-350**).
   - **Akciğer Parankimi İçin (Toraks):** Sol araç çubuğundaki Güneş/Kontrast ikonundan **Preset: Lung** seçin (veya elle **WL: -600, WW: 1500** yapın). Akciğer alanı şeffaflaşacak ve metastatik nodüller beyaz yuvarlak odak olarak belirginleşecektir.
2. Sol araç çubuğundan **Paintbrush (Fırça)** simgesini seçin *(Kısayol: `Alt + P`)*.
3. Aksiyel kesitte lezyon odaklarının içini boyayın:
   - **Boyut Ayarı:** Farenin sağ tuşuna basılı tutup sürükleyerek fırçayı büyütüp küçültebilirsiniz.
   - **Silme:** Farenin sağ tıkıyla veya sol paneldeki Silgi (`Alt + E`) ile düzeltebilirsiniz.

---

### 4. Adım: Kaydetme
- Etiketleme bittiğinde klavyeden **`Ctrl + S`** yapın *(veya **Segmentation ➔ Save Segmentation Image**)*.
- **Save / Overwrite** diyerek dosyanın üzerine kaydedin.

---

## 🩺 Klinik ve Pratik Püf Noktaları

1. **Seri Seçimi (En Kritik Kural):**
   - Karaciğer ve karın organları için **kesinlikle 2.5 mm Abdomen (ABD / Abdomen)** serisini açınız.
   - **Kemik penceresi (Toraks Kemik)** serisi üzerinden karaciğer/karın organları etiketlenmemelidir (kontrast ve yumuşak doku çözünürlüğü yanıltır).
   - Akciğer parankimi için ise Toraks serisi açılıp **Lung Preseti** (WL: -600, WW: 1500) kullanılmalıdır.

2. **Hastada Lezyon Yoksa (Remisyonda / Temiz İse):**
   - Karaciğer veya akciğer temizse hiçbir şey boyamadan maskeyi yine de **Ctrl + S** ile kaydediniz (içinde Label 8 olmayan boş maske dosyası).
   - *Neden önemli?* Yapay zeka eğitim motoru, hiç maske kaydedilmeyen serileri "incelenmemiş/etiketlenmemiş" sayarak eğitimden eler. Boş maske kaydedildiğinde ise sistem vakanın "doktor tarafından incelenip kesinlikle temiz olduğu teyit edilmiş" olduğunu anlar ve modele doğru negatiflik öğretir.

3. **Kist / Atelektazi / Granülom vs. Malignite:**
   - Basit karaciğer kistleri (<20 HU) ve benign kalsifiye granülomları hedef lezyon sayılmaz; bunları **etiketlemeyiniz**.
   - Kontrast tutan, solid, invaziv kenarlı primer veya metastatik kitle/nodülleri **Label 8 (Lesion)** ile etiketleyiniz.

4. **RECIST 1.1 Kuralı:**
   - RECIST 1.1'e göre hedef lezyonlar aksiyel en uzun çap üzerinden hesaplanır. Akciğer ve karaciğerdeki lezyonların aksiyel sınırlarının gerçeğe yakın çizilmesi yeterlidir.
