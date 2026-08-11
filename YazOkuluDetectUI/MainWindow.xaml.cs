using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using FellowOakDicom;
using Newtonsoft.Json;

namespace YazOkuluDetectUI
{
    public partial class MainWindow : Window
    {
        // FastAPI Backend Sunucu Adresi
        private const string ApiBaseUrl = "http://127.0.0.1:8000";
        private readonly HttpClient _httpClient = new HttpClient();

        private string _selectedDicomDir = string.Empty;
        private string _lastProcessedNifti = string.Empty;

        public MainWindow()
        {
            InitializeComponent();
            Log("Uygulama başlatıldı. FastAPI bağlantısı kontrol ediliyor...");
            _ = CheckApiConnectionAsync();
        }

        private void Log(string message)
        {
            string timestamp = DateTime.Now.ToString("HH:mm:ss");
            LstLogs.Items.Add($"[{timestamp}] {message}");
            LstLogs.ScrollIntoView(LstLogs.Items[LstLogs.Items.Count - 1]);
        }

        private async Task CheckApiConnectionAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{ApiBaseUrl}/");
                if (response.IsSuccessStatusCode)
                {
                    TxtApiStatus.Text = "Python API: Bağlı (http://127.0.0.1:8000)";
                    Log("FastAPI sunucusuyla bağlantı kuruldu.");
                }
                else
                {
                    TxtApiStatus.Text = "Python API: Yanıt Vermiyor";
                    Log("UYARI: FastAPI sunucusu yanıt vermedi.");
                }
            }
            catch (Exception ex)
            {
                TxtApiStatus.Text = "Python API: Bağlantı Hatası!";
                Log($"HATA: API sunucusuna erişilemedi. ({ex.Message})");
            }
        }

        private void BtnBrowseDicom_Click(object sender, RoutedEventArgs e)
        {
            // .NET 8 WPF yerel klasör seçici
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "DICOM Tetkik Klasörünü Seçin"
            };

            if (dialog.ShowDialog() == true)
            {
                _selectedDicomDir = dialog.FolderName;
                TxtDicomPath.Text = _selectedDicomDir;
                Log($"DICOM Klasörü seçildi: {_selectedDicomDir}");

                // fo-dicom ile klasördeki ilk DICOM dosyasının metadatasını oku
                ReadDicomMetadata(_selectedDicomDir);
            }
        }

        private void ReadDicomMetadata(string dirPath)
        {
            try
            {
                // Klasördeki tüm dosyaları (tüm alt klasörler dahil) tara
                var allFiles = Directory.GetFiles(dirPath, "*.*", SearchOption.AllDirectories)
                                        .Where(f => !f.EndsWith(".xml", StringComparison.OrdinalIgnoreCase) && 
                                                    !f.EndsWith(".txt", StringComparison.OrdinalIgnoreCase) &&
                                                    !f.EndsWith(".json", StringComparison.OrdinalIgnoreCase) &&
                                                    !Path.GetFileName(f).Equals("DICOMDIR", StringComparison.OrdinalIgnoreCase))
                                        .ToArray();

                bool foundValidDicom = false;

                foreach (var file in allFiles)
                {
                    try
                    {
                        var dicomFile = DicomFile.Open(file);
                        var dataset = dicomFile.Dataset;

                        string patId = dataset.GetSingleValueOrDefault(DicomTag.PatientID, "Bilinmiyor");
                        string patName = dataset.GetSingleValueOrDefault(DicomTag.PatientName, "Bilinmiyor");
                        string studyDate = dataset.GetSingleValueOrDefault(DicomTag.StudyDate, "Bilinmiyor");

                        TxtPatientId.Text = $"Hasta ID: {patId}";
                        TxtPatientName.Text = $"Hasta Adı: {patName}";
                        TxtStudyDate.Text = $"Çekim Tarihi: {studyDate}";

                        Log($"DICOM Üstveri okundu -> Hasta: {patName} ({patId}), Tarih: {studyDate}");
                        foundValidDicom = true;
                        break; // İlk geçerli DICOM dosyasını bulunca üstveriyi alıp çıkıyoruz
                    }
                    catch
                    {
                        // Bu dosya DICOM formatında değilse sonraki dosyayı dene
                        continue;
                    }
                }

                if (!foundValidDicom)
                {
                    TxtPatientId.Text = "Hasta ID: Klasör Seçildi";
                    TxtPatientName.Text = "Hasta Adı: (Klasör Algılandı)";
                    TxtStudyDate.Text = "Çekim Tarihi: -";
                    Log("DICOM Klasörü seçildi. (Ön İşleme adımı için hazırdır).");
                }
            }
            catch (Exception ex)
            {
                Log($"DICOM okuma uyarısı: {ex.Message}");
            }
        }

        private async void BtnPreprocess_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedDicomDir) || !Directory.Exists(_selectedDicomDir))
            {
                MessageBox.Show("Lütfen önce geçerli bir DICOM klasörü seçin.", "Uyarı", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            Log("Ön işleme isteği FastAPI'ye gönderiliyor...");
            BtnPreprocess.IsEnabled = false;

            try
            {
                string outputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Output");
                string outputFilename = "processed_study.nii.gz";

                var requestBody = new
                {
                    dicom_dir = _selectedDicomDir,
                    output_dir = outputDir,
                    output_filename = outputFilename
                };

                string jsonContent = JsonConvert.SerializeObject(requestBody);
                var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{ApiBaseUrl}/preprocess", content);
                string responseStr = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = Newtonsoft.Json.Linq.JObject.Parse(responseStr);
                    _lastProcessedNifti = result["file_path"]?.ToString() ?? string.Empty;
                    string spacing = result["spacing"]?.ToString() ?? "";
                    string size = result["size"]?.ToString() ?? "";

                    Log($"[BAŞARILI] Ön işleme tamamlandı: {_lastProcessedNifti}");
                    Log($"Resampled Spacing: {spacing}");

                    TxtReportOutput.Text = $"=== ÖN İŞLEME RAPORU ===\n\n" +
                                           $"Çıktı Dosyası: {_lastProcessedNifti}\n" +
                                           $"Hedef Spacing: [1.0mm, 1.0mm, 1.0mm]\n" +
                                           $"Görüntü Boyutları: {size}\n\n" +
                                           $"Görüntü başarıyla NIfTI formatına çevrildi ve normalizasyon uygulandı.";
                }
                else
                {
                    Log($"HATA: Preprocess isteği başarısız oldu. ({responseStr})");
                }
            }
            catch (Exception ex)
            {
                Log($"HATA: İletişim hatası. ({ex.Message})");
            }
            finally
            {
                BtnPreprocess.IsEnabled = true;
            }
        }

        private async void BtnPredict_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastProcessedNifti) || !File.Exists(_lastProcessedNifti))
            {
                MessageBox.Show("Lütfen önce 'DICOM Ön İşleme' adımını çalıştırın.", "Uyarı", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            Log("AI Segmentasyon tahmini (Mock) başlatılıyor...");
            BtnPredict.IsEnabled = false;

            try
            {
                string maskPath = _lastProcessedNifti.Replace(".nii.gz", "_mask.nii.gz");

                var requestBody = new
                {
                    nifti_path = _lastProcessedNifti,
                    output_mask_path = maskPath
                };

                string jsonContent = JsonConvert.SerializeObject(requestBody);
                var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{ApiBaseUrl}/predict-mock", content);
                string responseStr = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = Newtonsoft.Json.Linq.JObject.Parse(responseStr);
                    string maskResPath = result["mask_path"]?.ToString() ?? "";
                    string count = result["detected_lesions_count"]?.ToString() ?? "0";
                    string previewImgPath = result["preview_image_path"]?.ToString() ?? "";

                    Log($"[BAŞARILI] AI Maskesi Oluşturuldu: {maskResPath}");

                    if (!string.IsNullOrEmpty(previewImgPath) && File.Exists(previewImgPath))
                    {
                        LoadImageToViewer(previewImgPath);
                    }

                    TxtReportOutput.Text += $"\n\n=== AI SEGMENTASYON TAHMİNİ (MOCK) ===\n\n" +
                                           $"Segmentasyon Maskesi: {maskResPath}\n" +
                                           $"Tespit Edilen Hedef Lezyon Sayısı: {count}\n" +
                                           $"Lezyon Hacmi & Feret Çapı Hesaplandı.";
                }
                else
                {
                    Log($"HATA: Segmentasyon tahmini başarısız. ({responseStr})");
                }
            }
            catch (Exception ex)
            {
                Log($"HATA: {ex.Message}");
            }
            finally
            {
                BtnPredict.IsEnabled = true;
            }
        }

        private void LoadImageToViewer(string imagePath)
        {
            try
            {
                var bitmap = new System.Windows.Media.Imaging.BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                bitmap.EndInit();
                bitmap.Freeze();

                ImgSliceViewer.Source = bitmap;
                TxtImagePlaceholder.Visibility = Visibility.Collapsed;
                Log($"[GÖRSEL] 2D BT Kesiti ve AI Lezyon Etiketleri Yüklendi.");
            }
            catch (Exception ex)
            {
                Log($"Görsel yükleme hatası: {ex.Message}");
            }
        }

        private async void BtnRunRecist_Click(object sender, RoutedEventArgs e)
        {
            if (!double.TryParse(TxtBaselineSod.Text, out double baselineSod) ||
                !double.TryParse(TxtFollowupSod.Text, out double followupSod))
            {
                MessageBox.Show("Lütfen geçerli sayısal SOD değerleri girin.", "Hata", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            bool newLesion = ChkNewLesion.IsChecked ?? false;

            Log("RECIST 1.1 Karar Motoru çalıştırılıyor...");

            try
            {
                var requestBody = new
                {
                    sod_baseline = baselineSod,
                    sod_followup = followupSod,
                    new_lesion = newLesion
                };

                string jsonContent = JsonConvert.SerializeObject(requestBody);
                var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{ApiBaseUrl}/recist-decision", content);
                string responseStr = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = Newtonsoft.Json.Linq.JObject.Parse(responseStr);

                    string decision = result["decision"]?.ToString() ?? "SD";
                    string explanation = result["explanation"]?.ToString() ?? "";
                    double pct = result["change_percentage"]?.ToObject<double>() ?? 0.0;

                    // UI Güncelle
                    TxtDecisionBadge.Text = decision;
                    TxtDecisionExplanation.Text = explanation;
                    TxtBaselineSummary.Text = $"Baz SOD: {baselineSod:F1} mm";
                    TxtFollowupSummary.Text = $"Kontrol SOD: {followupSod:F1} mm";
                    TxtChangeSummary.Text = $"Değişim: {(pct >= 0 ? "+" : "")}{pct:F1}%";

                    // Rozet Rengini Ayarla
                    switch (decision)
                    {
                        case "CR":
                            BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#10B981")); // Yeşil
                            TxtDecisionTitle.Text = "Complete Response (Tam Yanıt)";
                            break;
                        case "PR":
                            BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0284C7")); // Mavi
                            TxtDecisionTitle.Text = "Partial Response (Kısmi Yanıt)";
                            break;
                        case "SD":
                            BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#EAB308")); // Sarı
                            TxtDecisionTitle.Text = "Stable Disease (Stabil Hastalık)";
                            break;
                        case "PD":
                            BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#EF4444")); // Kırmızı
                            TxtDecisionTitle.Text = "Progressive Disease (İlerleyen Hastalık)";
                            break;
                    }

                    Log($"[KARAR] RECIST 1.1 Sonucu: {decision} (%{pct:F1})");

                    // Rapor Paneline Yaz
                    TxtReportOutput.Text = $"====================================================\n" +
                                           $"     ONKOLOJİK BT RECIST 1.1 KARAR DESTEK RAPORU    \n" +
                                           $"====================================================\n\n" +
                                           $"KLİNİK DEĞERLENDİRME: {decision}\n" +
                                           $"AÇIKLAMA: {explanation}\n\n" +
                                           $"ÖLÇÜM DETAYLARI:\n" +
                                           $" - Baz (t0) Toplam Tümör Çapı (SOD): {baselineSod:F1} mm\n" +
                                           $" - Kontrol (t1) Toplam Tümör Çapı (SOD): {followupSod:F1} mm\n" +
                                           $" - Tümör Yükü Değişimi: %{pct:F1}\n" +
                                           $" - Yeni Lezyon Varlığı: {(newLesion ? "EVET" : "HAYIR")}\n\n" +
                                           $"SİSTEM NOTU: Bu karar tamamen RECIST 1.1 deterministik kural motoru tarafından üretilmiştir.";
                }
            }
            catch (Exception ex)
            {
                Log($"HATA: {ex.Message}");
            }
        }
    }
}