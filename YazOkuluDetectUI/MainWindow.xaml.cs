using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using FellowOakDicom;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace YazOkuluDetectUI
{
    public partial class MainWindow : Window
    {
        // ─── SABİTLER ────────────────────────────────────────────────────────
        private const string ApiBaseUrl = "http://127.0.0.1:8000";

        // ─── DURUM DEĞİŞKENLERİ ─────────────────────────────────────────────
        private readonly HttpClient _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };

        // Mevcut mod: "single" veya "longitudinal"
        private string _mode = "single";

        // Tek tetkik modu için seçili klasör
        private string _singleDicomDir = string.Empty;

        // Longitudinal mod için seçili klasörler
        private string _baselineDicomDir  = string.Empty;
        private string _followupDicomDir  = string.Empty;

        // t0 ve t1 Hasta ID'leri (aynı hasta doğrulaması için)
        private string _baselinePatientId = string.Empty;
        private string _followupPatientId = string.Empty;

        // ─── KURUCU ─────────────────────────────────────────────────────────
        public MainWindow()
        {
            InitializeComponent();
            Log("Uygulama başlatıldı. FastAPI backend bağlantısı kontrol ediliyor...");
            _ = CheckApiConnectionAsync();
        }

        // ====================================================================
        //  YARDIMCI: LOG
        // ====================================================================
        private void Log(string message)
        {
            string ts = DateTime.Now.ToString("HH:mm:ss");
            LstLogs.Items.Add($"[{ts}] {message}");
            LstLogs.ScrollIntoView(LstLogs.Items[LstLogs.Items.Count - 1]);
        }

        // ====================================================================
        //  API BAĞLANTI KONTROLÜ
        // ====================================================================
        private async Task CheckApiConnectionAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{ApiBaseUrl}/");
                if (response.IsSuccessStatusCode)
                {
                    TxtApiStatus.Text = "Python API: Bağlı (http://127.0.0.1:8000)";
                    Log("FastAPI sunucusuyla bağlantı kuruldu.");

                    // Model ağırlığı var mı kontrol et
                    await CheckAiModeAsync();
                }
                else
                {
                    TxtApiStatus.Text = "Python API: Yanıt Vermiyor";
                    BadgeApi.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#7F1D1D"));
                    EllApiDot.Fill = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#EF4444"));
                    TxtApiStatus.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FCA5A5"));
                    Log("UYARI: FastAPI sunucusu yanıt vermedi. Backend'i başlatın!");
                }
            }
            catch (Exception ex)
            {
                TxtApiStatus.Text = "Python API: Bağlantı Hatası!";
                BadgeApi.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#7F1D1D"));
                EllApiDot.Fill = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#EF4444"));
                TxtApiStatus.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FCA5A5"));
                Log($"HATA: API sunucusuna erişilemedi. ({ex.Message})");
            }
        }

        /// <summary>Modeller klasöründe eğitilmiş ağırlık dosyası var mı kontrol eder ve AI modu rozetini günceller.</summary>
        private async Task CheckAiModeAsync()
        {
            // ./models klasörünü farklı olası geliştirme/çalışma yollarında ara
            string[] candidateDirs = new[]
            {
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models"),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "models"),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models"),
                Path.Combine(Directory.GetCurrentDirectory(), "models")
            };
            bool hasWeights = candidateDirs.Any(d => Directory.Exists(d) &&
                              (Directory.GetFiles(d, "*.pth", SearchOption.AllDirectories).Length > 0 ||
                               Directory.GetFiles(d, "*.pt", SearchOption.AllDirectories).Length > 0));

            await Dispatcher.InvokeAsync(() =>
            {
                if (hasWeights)
                {
                    TxtAiMode.Text = "Gerçek AI Modu (nnU-Net)";
                    TxtAiModeIcon.Text = "🧠";
                    BadgeAiMode.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#064E3B"));
                    TxtAiMode.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#34D399"));
                }
                else
                {
                    TxtAiMode.Text = "Test Modu (Model Eğitilmemiş)";
                    TxtAiModeIcon.Text = "🔧";
                }
            });
        }

        // ====================================================================
        //  MOD SEÇİCİ
        // ====================================================================
        private void BtnModeSingle_Click(object sender, RoutedEventArgs e)
        {
            SetMode("single");
        }

        private void BtnModeLongitudinal_Click(object sender, RoutedEventArgs e)
        {
            SetMode("longitudinal");
        }

        private void SetMode(string mode)
        {
            _mode = mode;

            if (mode == "single")
            {
                BtnModeSingle.Style       = (Style)FindResource("TabBtnActive");
                BtnModeLongitudinal.Style = (Style)FindResource("TabBtn");
                PanelSingle.Visibility       = Visibility.Visible;
                PanelLongitudinal.Visibility = Visibility.Collapsed;
                TxtModeDescription.Text = "Tek bir BT taraması için lezyon tespiti ve sınıflandırma yapın.";
                BtnAnalyze.Content = "🔬  Analizi Başlat";
                UpdateAnalyzeButtonState();
            }
            else
            {
                BtnModeSingle.Style       = (Style)FindResource("TabBtn");
                BtnModeLongitudinal.Style = (Style)FindResource("TabBtnActive");
                PanelSingle.Visibility       = Visibility.Collapsed;
                PanelLongitudinal.Visibility = Visibility.Visible;
                TxtModeDescription.Text = "İki farklı tarihteki BT taramalarını karşılaştırarak RECIST 1.1 kararı üretin.";
                BtnAnalyze.Content = "📊  Longitudinal Analizi Başlat";
                UpdateAnalyzeButtonState();
            }
        }

        // ====================================================================
        //  KLASÖR GÖZAT ve DICOM METADATA OKUMA
        // ====================================================================
        private void BtnBrowseSingle_Click(object sender, RoutedEventArgs e)
        {
            var folder = BrowseFolder("BT Tetkik DICOM Klasörünü Seçin");
            if (folder == null) return;
            _singleDicomDir = folder;
            TxtDicomPath.Text = folder;
            var meta = ReadDicomMetadata(folder);
            TxtSinglePatientId.Text   = $"Hasta ID: {meta.Id}";
            TxtSinglePatientName.Text = $"Hasta Adı: {meta.Name}";
            TxtSingleStudyDate.Text   = $"Çekim Tarihi: {meta.Date}";
            Log($"Tek Tetkik: {meta.Name} ({meta.Id}), Tarih: {meta.Date}");
            UpdateAnalyzeButtonState();
        }

        private void BtnBrowseBaseline_Click(object sender, RoutedEventArgs e)
        {
            var folder = BrowseFolder("Baseline (t₀) DICOM Klasörünü Seçin");
            if (folder == null) return;
            _baselineDicomDir = folder;
            TxtBaselinePath.Text = folder;
            var meta = ReadDicomMetadata(folder);
            _baselinePatientId = meta.Id;
            TxtBaselinePatientId.Text   = $"Hasta ID: {meta.Id}";
            TxtBaselinePatientName.Text = $"Hasta Adı: {meta.Name}";
            TxtBaselineStudyDate.Text   = $"Çekim Tarihi: {meta.Date}";
            Log($"t₀ Baseline: {meta.Name} ({meta.Id}), Tarih: {meta.Date}");
            ValidatePatientMatch();
            UpdateAnalyzeButtonState();
        }

        private void BtnBrowseFollowup_Click(object sender, RoutedEventArgs e)
        {
            var folder = BrowseFolder("Follow-up (t₁) DICOM Klasörünü Seçin");
            if (folder == null) return;
            _followupDicomDir = folder;
            TxtFollowupPath.Text = folder;
            var meta = ReadDicomMetadata(folder);
            _followupPatientId = meta.Id;
            TxtFollowupPatientId.Text   = $"Hasta ID: {meta.Id}";
            TxtFollowupPatientName.Text = $"Hasta Adı: {meta.Name}";
            TxtFollowupStudyDate.Text   = $"Çekim Tarihi: {meta.Date}";
            Log($"t₁ Follow-up: {meta.Name} ({meta.Id}), Tarih: {meta.Date}");
            ValidatePatientMatch();
            UpdateAnalyzeButtonState();
        }

        /// <summary>Klasör seçici dialog'u açar ve seçilen yolu döner. İptal edilirse null döner.</summary>
        private string? BrowseFolder(string title)
        {
            var dialog = new Microsoft.Win32.OpenFolderDialog { Title = title };
            return (dialog.ShowDialog() == true) ? dialog.FolderName : null;
        }

        // ====================================================================
        //  HASTA KİMLİĞİ DOĞRULAMASI (LONGİTUDİNAL MOD)
        // ====================================================================
        private void ValidatePatientMatch()
        {
            if (string.IsNullOrEmpty(_baselinePatientId) || string.IsNullOrEmpty(_followupPatientId))
            {
                BorderPatientMismatch.Visibility = Visibility.Collapsed;
                BorderPatientMatch.Visibility    = Visibility.Collapsed;
                return;
            }

            bool match = string.Equals(_baselinePatientId.Trim(), _followupPatientId.Trim(),
                                       StringComparison.OrdinalIgnoreCase);

            if (match)
            {
                BorderPatientMismatch.Visibility = Visibility.Collapsed;
                BorderPatientMatch.Visibility    = Visibility.Visible;
                TxtPatientMatch.Text = $"✅ Aynı hasta doğrulandı (ID: {_baselinePatientId}). Longitudinal analiz hazır.";
                Log($"Hasta ID eşleşmesi doğrulandı: {_baselinePatientId}");
            }
            else
            {
                BorderPatientMatch.Visibility    = Visibility.Collapsed;
                BorderPatientMismatch.Visibility = Visibility.Visible;
                TxtPatientMismatch.Text = $"⚠️ Hasta ID uyuşmuyor! t₀: [{_baselinePatientId}] — t₁: [{_followupPatientId}]. Devam etmek istiyor musunuz?";
                Log($"UYARI: Hasta ID uyuşmazlığı! t0={_baselinePatientId}, t1={_followupPatientId}");
            }
        }

        // ====================================================================
        //  ANALİZ BUTONU AKTİFLİĞİ
        // ====================================================================
        private void UpdateAnalyzeButtonState()
        {
            BtnAnalyze.IsEnabled = _mode == "single"
                ? !string.IsNullOrEmpty(_singleDicomDir)
                : !string.IsNullOrEmpty(_baselineDicomDir) && !string.IsNullOrEmpty(_followupDicomDir);
        }

        // ====================================================================
        //  ANA ANALİZ BUTONU
        // ====================================================================
        private async void BtnAnalyze_Click(object sender, RoutedEventArgs e)
        {
            BtnAnalyze.IsEnabled = false;
            BorderProgress.Visibility = Visibility.Visible;
            ResetResultPanel();

            try
            {
                if (_mode == "single")
                    await RunSinglePipelineAsync();
                else
                    await RunLongitudinalPipelineAsync();
            }
            finally
            {
                BtnAnalyze.IsEnabled = true;
                BorderProgress.Visibility = Visibility.Collapsed;
                SetFooterStatus("Analiz tamamlandı.");
            }
        }

        // ====================================================================
        //  TEK TEKTİK PİPELINE  →  POST /pipeline-single
        // ====================================================================
        private async Task RunSinglePipelineAsync()
        {
            Log("─── Tek Tetkik Pipeline Başlatıldı ───");
            SetFooterStatus("Analiz çalışıyor...");

            SetProgress(5, "Adım 1/3: DICOM ön işleme başlıyor...");

            // DICOM metadata'dan hasta bilgilerini al
            var meta = ReadDicomMetadata(_singleDicomDir);

            string outputDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "RECIST_Output", $"{meta.Id}_{DateTime.Now:yyyyMMdd_HHmmss}");

            var payload = new
            {
                dicom_dir    = _singleDicomDir,
                output_dir   = outputDir,
                patient_id   = meta.Id,
                patient_name = meta.Name,
                study_date   = meta.Date
            };

            SetProgress(20, "Adım 2/3: AI Segmentasyon çalışıyor...");

            try
            {
                string json = JsonConvert.SerializeObject(payload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                var response = await _httpClient.PostAsync($"{ApiBaseUrl}/pipeline-single", content);
                string responseStr = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    Log($"HATA: Pipeline isteği başarısız. ({responseStr})");
                    SetProgress(0, "Hata oluştu.");
                    return;
                }

                SetProgress(90, "Adım 3/3: Rapor oluşturuluyor...");

                var result = JObject.Parse(responseStr);
                DisplaySingleResults(result, meta);
                SetProgress(100, "Tamamlandı!");
                Log("─── Tek Tetkik Pipeline Tamamlandı ───");
            }
            catch (Exception ex)
            {
                Log($"HATA: {ex.Message}");
                SetProgress(0, "Hata oluştu.");
            }
        }

        // ====================================================================
        //  LONGİTUDİNAL PİPELINE  →  POST /pipeline
        // ====================================================================
        private async Task RunLongitudinalPipelineAsync()
        {
            Log("─── Longitudinal Pipeline Başlatıldı ───");
            SetFooterStatus("Longitudinal analiz çalışıyor...");

            SetProgress(5, "Adım 1/6: t₀ + t₁ DICOM ön işleme...");

            var metaBaseline = ReadDicomMetadata(_baselineDicomDir);
            var metaFollowup = ReadDicomMetadata(_followupDicomDir);

            string outputDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "RECIST_Output", $"{metaBaseline.Id}_longitudinal_{DateTime.Now:yyyyMMdd_HHmmss}");

            var payload = new
            {
                baseline_dicom_dir = _baselineDicomDir,
                followup_dicom_dir = _followupDicomDir,
                output_dir         = outputDir,
                patient_id         = metaBaseline.Id,
                patient_name       = metaBaseline.Name,
                study_date         = metaBaseline.Date,
                use_llm            = false
            };

            SetProgress(15, "Adım 2/6: t₀ AI Segmentasyon...");

            try
            {
                string json = JsonConvert.SerializeObject(payload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                // Uzun sürebileceği için progress güncelle (API tamamlandığında iptal edilir)
                using var cts = new CancellationTokenSource();
                var progressTask = SimulateProgressAsync(15, 85, 14, cts.Token);
                var response = await _httpClient.PostAsync($"{ApiBaseUrl}/pipeline", content);
                cts.Cancel();
                try { await progressTask; } catch { }

                string responseStr = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    Log($"HATA: Longitudinal pipeline isteği başarısız. ({responseStr})");
                    SetProgress(0, "Hata oluştu.");
                    return;
                }

                SetProgress(90, "Adım 6/6: Rapor oluşturuluyor...");
                var result = JObject.Parse(responseStr);
                DisplayLongitudinalResults(result, metaBaseline);
                SetProgress(100, "Tamamlandı!");
                Log("─── Longitudinal Pipeline Tamamlandı ───");
            }
            catch (Exception ex)
            {
                Log($"HATA: {ex.Message}");
                SetProgress(0, "Hata oluştu.");
            }
        }

        /// <summary>Progress bar'ı başlangıç değerinden bitiş değerine ilerleten animasyon görevidir.</summary>
        private async Task SimulateProgressAsync(int from, int to, int steps, CancellationToken ct = default)
        {
            int step = Math.Max(1, (to - from) / steps);
            for (int i = from; i < to; i += step)
            {
                if (ct.IsCancellationRequested) break;
                try { await Task.Delay(1000, ct); } catch { break; }
                if (ct.IsCancellationRequested) break;
                int current = Math.Min(i + step, to);
                Dispatcher.Invoke(() => SetProgress(current, $"Analiz devam ediyor... (%{current})"));
            }
        }

        // ====================================================================
        //  SONUÇ GÖSTERİM: TEK TEKTİK
        // ====================================================================
        private void DisplaySingleResults(JObject result, DicomMeta meta)
        {
            try
            {
                // Lezyon metrikleri (Hem flat hem nested anahtar desteği)
                var singleResult = result["single_timepoint_result"] ?? result;
                var seg = singleResult["segmentation"];
                int lesionCount  = singleResult["lesion_count"]?.ToObject<int>() ??
                                   seg?["detected_lesions_count"]?.ToObject<int>() ??
                                   singleResult["radiomics"]?["total_candidate_regions"]?.ToObject<int>() ?? 0;
                double totalVol  = singleResult["total_volume_mm3"]?.ToObject<double>() ??
                                   seg?["total_volume_mm3"]?.ToObject<double>() ?? 0;
                double sod       = singleResult["sod_mm"]?.ToObject<double>() ??
                                   singleResult["radiomics"]?["sod_mm"]?.ToObject<double>() ?? 0;

                // Karar kartı (tek tetkikte RECIST yok, "Baseline Tespit" göster)
                TxtDecisionBadge.Text    = "BL";
                TxtDecisionTitle.Text    = $"Baseline Tetkik — {meta.Name}";
                TxtDecisionExplanation.Text = $"Hasta ID: {meta.Id} · Çekim: {meta.Date}";
                BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1D4ED8"));

                TxtBaselineSummary.Text  = $"SOD: {sod:F1} mm";
                TxtFollowupSummary.Text  = "";
                TxtChangeSummary.Text    = "";
                TxtLesionCount.Text      = lesionCount.ToString();
                TxtVolumeSummary.Text    = $"{totalVol:F0} mm³";

                // Rapor
                string report = result["report"]?["report_text"]?.ToString() ??
                                result["clinical_report"]?.ToString() ??
                                BuildFallbackSingleReport(meta, lesionCount, sod, totalVol);
                TxtReportOutput.Text = report;
                Log($"[SONUÇ] Baseline — {lesionCount} lezyon, SOD: {sod:F1} mm");

                // Preview görsel
                string previewPath = singleResult["preview_image_path"]?.ToString() ??
                                     seg?["preview_image_path"]?.ToString() ??
                                     singleResult["radiomics"]?["preview_image_path"]?.ToString() ?? "";
                if (!string.IsNullOrEmpty(previewPath) && File.Exists(previewPath))
                    LoadImageToViewer(previewPath);
            }
            catch (Exception ex)
            {
                Log($"Sonuç gösterme hatası: {ex.Message}");
            }
        }

        // ====================================================================
        //  SONUÇ GÖSTERİM: LONGİTUDİNAL
        // ====================================================================
        private void DisplayLongitudinalResults(JObject result, DicomMeta meta)
        {
            try
            {
                // RECIST kararı
                var recist = result["recist_decision"] ?? result;
                string decision    = recist["decision"]?.ToString()          ?? "SD";
                string explanation = recist["explanation"]?.ToString()       ?? "";
                double sodBl       = recist["baseline_sod"]?.ToObject<double>() ??
                                     result["matching"]?["sod_baseline"]?.ToObject<double>() ?? 0;
                double sodFu       = recist["followup_sod"]?.ToObject<double>() ??
                                     result["matching"]?["sod_followup"]?.ToObject<double>() ?? 0;
                double changePct   = recist["change_percentage"]?.ToObject<double>() ??
                                     result["matching"]?["change_percentage"]?.ToObject<double>() ?? 0;
                bool   newLesion   = recist["new_lesion"]?.ToObject<bool>() ?? false;

                int lesionCount    = result["matching"]?["matched_lesions"]?.ToObject<int>() ??
                                     result["longitudinal"]?["matching"]?["matched_pairs"]?.Count() ??
                                     result["segmentation"]?["followup"]?["detected_lesions_count"]?.ToObject<int>() ??
                                     result["segmentation_followup"]?["detected_lesions_count"]?.ToObject<int>() ?? 0;

                // Karar rozeti rengi ve başlık
                TxtDecisionBadge.Text    = decision;
                TxtDecisionExplanation.Text = explanation;
                TxtBaselineSummary.Text  = $"t₀ SOD: {sodBl:F1} mm";
                TxtFollowupSummary.Text  = $"t₁ SOD: {sodFu:F1} mm";

                string changeSign = changePct >= 0 ? "+" : "";
                TxtChangeSummary.Text = $"Δ {changeSign}{changePct:F1}%";
                TxtLesionCount.Text   = lesionCount.ToString();

                switch (decision)
                {
                    case "CR":
                        BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#065F46"));
                        TxtDecisionTitle.Text    = "Complete Response (Tam Yanıt) ✅";
                        TxtChangeSummary.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#34D399"));
                        break;
                    case "PR":
                        BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1D4ED8"));
                        TxtDecisionTitle.Text    = "Partial Response (Kısmi Yanıt)";
                        TxtChangeSummary.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#60A5FA"));
                        break;
                    case "SD":
                        BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#78350F"));
                        TxtDecisionTitle.Text    = "Stable Disease (Stabil)";
                        TxtChangeSummary.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FCD34D"));
                        break;
                    case "PD":
                        BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#7F1D1D"));
                        TxtDecisionTitle.Text    = "Progressive Disease (İlerleme) ⚠️";
                        TxtChangeSummary.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F87171"));
                        break;
                }

                // Rapor
                string report = result["report"]?["report_text"]?.ToString() ??
                                result["clinical_report"]?.ToString() ??
                                BuildFallbackLongitudinalReport(meta, decision, explanation, sodBl, sodFu, changePct, newLesion);
                TxtReportOutput.Text = report;
                Log($"[KARAR] {decision} — SOD: {sodBl:F1}→{sodFu:F1} mm, Δ{changeSign}{changePct:F1}%");

                // Preview görsel (Longitudinal yan yana veya followup önizlemesi)
                string previewPath = result["preview_image_path"]?.ToString() ??
                                     result["segmentation"]?["followup"]?["preview_image_path"]?.ToString() ??
                                     result["segmentation_followup"]?["preview_image_path"]?.ToString() ??
                                     result["segmentation"]?["baseline"]?["preview_image_path"]?.ToString() ??
                                     result["segmentation_baseline"]?["preview_image_path"]?.ToString() ?? "";
                if (!string.IsNullOrEmpty(previewPath) && File.Exists(previewPath))
                    LoadImageToViewer(previewPath);
            }
            catch (Exception ex)
            {
                Log($"Sonuç gösterme hatası: {ex.Message}");
            }
        }

        // ====================================================================
        //  YARDIMCI: GÖRSEL YÜKLEYİCİ
        // ====================================================================
        private void LoadImageToViewer(string imagePath)
        {
            try
            {
                var bitmap = new System.Windows.Media.Imaging.BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                bitmap.UriSource   = new Uri(imagePath, UriKind.Absolute);
                bitmap.EndInit();
                bitmap.Freeze();
                ImgSliceViewer.Source = bitmap;
                TxtImagePlaceholder.Visibility = Visibility.Collapsed;
                Log($"[GÖRSEL] BT Kesiti yüklendi: {Path.GetFileName(imagePath)}");
            }
            catch (Exception ex)
            {
                Log($"Görsel yükleme hatası: {ex.Message}");
            }
        }

        // ====================================================================
        //  YARDIMCI: İLERLEME ÇUBUĞU
        // ====================================================================
        private void SetProgress(int value, string stepText)
        {
            Dispatcher.Invoke(() =>
            {
                PbProgress.Value     = value;
                TxtProgressStep.Text = stepText;
                TxtProgressPct.Text  = $"{value}%";
                SetFooterStatus(stepText);
            });
        }

        private void SetFooterStatus(string status)
        {
            Dispatcher.Invoke(() => TxtFooterStatus.Text = $"Durum: {status}");
        }

        // ====================================================================
        //  YARDIMCI: SONUÇ PANELİNİ SIFIRLA
        // ====================================================================
        private void ResetResultPanel()
        {
            TxtDecisionBadge.Text    = "...";
            TxtDecisionTitle.Text    = "Analiz çalışıyor...";
            TxtDecisionExplanation.Text = "Lütfen bekleyin.";
            BadgeDecision.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E3A5F"));
            TxtBaselineSummary.Text  = "";
            TxtFollowupSummary.Text  = "";
            TxtChangeSummary.Text    = "";
            TxtLesionCount.Text      = "...";
            TxtVolumeSummary.Text    = "";
            TxtReportOutput.Text     = "Analiz devam ediyor. Lütfen bekleyin...";
            ImgSliceViewer.Source    = null;
            TxtImagePlaceholder.Visibility = Visibility.Visible;
        }

        // ====================================================================
        //  YARDIMCI: DICOM METADATA OKUMA
        // ====================================================================
        private DicomMeta ReadDicomMetadata(string dirPath)
        {
            var result = new DicomMeta { Id = "Bilinmiyor", Name = "Bilinmiyor", Date = "—" };
            try
            {
                var files = Directory.GetFiles(dirPath, "*.*", SearchOption.AllDirectories)
                    .Where(f => !f.EndsWith(".xml", StringComparison.OrdinalIgnoreCase)
                             && !f.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                             && !f.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                             && !Path.GetFileName(f).Equals("DICOMDIR", StringComparison.OrdinalIgnoreCase));

                foreach (var file in files)
                {
                    try
                    {
                        var ds = DicomFile.Open(file).Dataset;
                        result.Id   = ds.GetSingleValueOrDefault(DicomTag.PatientID,   "Bilinmiyor");
                        result.Name = ds.GetSingleValueOrDefault(DicomTag.PatientName, "Bilinmiyor");
                        result.Date = ds.GetSingleValueOrDefault(DicomTag.StudyDate,   "—");
                        // Tarihi okunabilir formata çevir: 20240115 → 15.01.2024
                        if (result.Date.Length == 8 && long.TryParse(result.Date, out _))
                            result.Date = $"{result.Date[6..8]}.{result.Date[4..6]}.{result.Date[0..4]}";
                        break;
                    }
                    catch { /* DICOM olmayan dosyalar atla */ }
                }
            }
            catch (Exception ex) { Log($"DICOM okuma uyarısı: {ex.Message}"); }
            return result;
        }

        // ====================================================================
        //  YARDIMCI: YEDEKLEMİ RAPOR ŞABLONLARİ (Backend rapor dönmezse)
        // ====================================================================
        private string BuildFallbackSingleReport(DicomMeta meta, int count, double sod, double vol)
        {
            return $"""
                ============================================================
                     ONKOLOJİK BT ANALİZ RAPORU — BASELINE TEKTİK
                ============================================================

                HASTA BİLGİLERİ:
                 - Hasta ID   : {meta.Id}
                 - Hasta Adı  : {meta.Name}
                 - Çekim Tarihi: {meta.Date}

                LEZYON TESPİT ÖZETI:
                 - Tespit Edilen Lezyon Sayısı  : {count}
                 - Toplam Tümör Çap Toplamı (SOD): {sod:F1} mm
                 - Toplam Lezyon Hacmi           : {vol:F0} mm³

                KLİNİK DEĞERLENDIRME:
                 Bu tetkik ilk / referans (Baseline) çekim olarak kaydedilmiştir.
                 Longitudinal karşılaştırma için Follow-up tetkiki beklenmektedir.

                SİSTEM NOTU:
                 Bu değerlendirme RECIST 1.1 deterministik kural motoru tarafından
                 üretilmiştir. LLM hiçbir klinik karar vermemiştir.
                """;
        }

        private string BuildFallbackLongitudinalReport(DicomMeta meta, string decision,
            string explanation, double sodBl, double sodFu, double pct, bool newLesion)
        {
            string sign = pct >= 0 ? "+" : "";
            return $"""
                ============================================================
                   ONKOLOJİK BT RECIST 1.1 LONGİTUDİNAL ANALİZ RAPORU
                ============================================================

                HASTA BİLGİLERİ:
                 - Hasta ID  : {meta.Id}
                 - Hasta Adı : {meta.Name}
                 - Referans Tarihi: {meta.Date}

                KLİNİK DEĞERLENDİRME: {decision}
                AÇIKLAMA: {explanation}

                ÖLÇÜM DETAYLARI:
                 - Baseline (t₀) Toplam Tümör Çapı (SOD): {sodBl:F1} mm
                 - Follow-up (t₁) Toplam Tümör Çapı (SOD): {sodFu:F1} mm
                 - Tümör Yükü Değişimi                   : {sign}{pct:F1}%
                 - Yeni Lezyon Varlığı                   : {(newLesion ? "EVET ⚠️" : "HAYIR")}

                SİSTEM NOTU:
                 Bu karar tamamen RECIST 1.1 deterministik kural motoru tarafından
                 üretilmiştir. LLM hiçbir klinik karar vermemiştir.
                """;
        }

        // ====================================================================
        //  İÇ SINIF: DICOM META VERISI
        // ====================================================================
        private class DicomMeta
        {
            public string Id   { get; set; } = "Bilinmiyor";
            public string Name { get; set; } = "Bilinmiyor";
            public string Date { get; set; } = "—";
        }
    }
}