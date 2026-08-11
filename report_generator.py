import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

def generate_clinical_report(pipeline_results: Dict[str, Any],
                              patient_info: Optional[Dict[str, str]] = None,
                              use_llm: bool = False,
                              llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Tüm pipeline çıktılarını (segmentasyon, radiomics, longitudinal, RECIST)
    standart onkolojik radyoloji rapor formatına dönüştürür.
    
    ÖNEMLİ TASARIM KARARI:
    ========================
    LLM (Gemini / GPT-4o) bu sistemde ASLA tıbbi karar VERMEZ.
    LLM'in TEK görevi: Deterministik kural motoru tarafından ÜRETİLMİŞ ve
    DOĞRULANMIŞ sayısal verileri, standart radyoloji raporu diline çevirmektir.
    
    Eğer use_llm=False ise (veya API key yoksa), şablon tabanlı rapor üretilir.
    Şablon tabanlı rapor da aynı klinik bilgiyi içerir, sadece dil akışı daha kalıptır.
    
    Args:
        pipeline_results: Tüm pipeline aşamalarının sonuçları
        patient_info: Hasta bilgileri (ID, Ad, Tarih) — opsiyonel
        use_llm: Gemini/GPT-4o API kullanılsın mı?
        llm_api_key: API anahtarı
        
    Returns:
        Rapor metni (Türkçe + İngilizce) ve metadata
    """
    # Varsayılan hasta bilgileri
    if patient_info is None:
        patient_info = {
            "patient_id": "Bilinmiyor",
            "patient_name": "Bilinmiyor",
            "study_date": datetime.now().strftime("%Y-%m-%d")
        }
    
    # Pipeline sonuçlarını güvenli bir şekilde çıkar
    recist = pipeline_results.get("recist_decision", {})
    segmentation = pipeline_results.get("segmentation", {})
    radiomics = pipeline_results.get("radiomics", {})
    longitudinal = pipeline_results.get("longitudinal", {})
    matching = longitudinal.get("matching", {})
    recist_input = longitudinal.get("recist_input", {})
    
    decision = recist.get("decision", "N/A")
    explanation = recist.get("explanation", "")
    change_pct = recist.get("change_percentage", 0.0)
    sod_baseline = recist.get("baseline_sod", recist_input.get("sod_baseline", 0.0))
    sod_followup = recist.get("followup_sod", recist_input.get("sod_followup", 0.0))
    new_lesion = recist.get("new_lesion", recist_input.get("new_lesion", False))
    
    lesion_count_bl = longitudinal.get("baseline_lesion_count", segmentation.get("detected_lesions_count", 0))
    lesion_count_fu = longitudinal.get("followup_lesion_count", 0)
    matched_pairs = matching.get("matched_pairs", [])
    new_lesions = matching.get("new_lesions", [])
    disappeared = matching.get("disappeared_lesions", [])
    
    regions = radiomics.get("regions", [])
    needs_review = radiomics.get("requires_radiologist_review", False)
    
    is_mock = segmentation.get("is_mock", True)
    
    # Rapor üretim zamanı
    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if use_llm and llm_api_key:
        report_text = _generate_llm_report(
            decision=decision, explanation=explanation, change_pct=change_pct,
            sod_baseline=sod_baseline, sod_followup=sod_followup,
            new_lesion=new_lesion, lesion_count_bl=lesion_count_bl,
            lesion_count_fu=lesion_count_fu, matched_pairs=matched_pairs,
            new_lesions=new_lesions, disappeared=disappeared,
            regions=regions, needs_review=needs_review,
            patient_info=patient_info, api_key=llm_api_key
        )
        report_source = "LLM (Gemini API)"
    else:
        report_text = _generate_template_report(
            decision=decision, explanation=explanation, change_pct=change_pct,
            sod_baseline=sod_baseline, sod_followup=sod_followup,
            new_lesion=new_lesion, lesion_count_bl=lesion_count_bl,
            lesion_count_fu=lesion_count_fu, matched_pairs=matched_pairs,
            new_lesions=new_lesions, disappeared=disappeared,
            regions=regions, needs_review=needs_review,
            patient_info=patient_info, is_mock=is_mock
        )
        report_source = "Şablon Tabanlı (Template)"
    
    return {
        "status": "success",
        "report_text": report_text,
        "report_source": report_source,
        "generated_at": report_timestamp,
        "recist_decision": decision,
        "requires_radiologist_review": needs_review,
        "is_ai_mock": is_mock
    }


def _generate_template_report(decision, explanation, change_pct, sod_baseline, sod_followup,
                                new_lesion, lesion_count_bl, lesion_count_fu,
                                matched_pairs, new_lesions, disappeared,
                                regions, needs_review, patient_info, is_mock) -> str:
    """
    LLM olmadan, tamamen şablon tabanlı standart onkolojik radyoloji raporu üretir.
    Klinik içerik olarak LLM versiyonuyla eşdeğerdir.
    """
    # Karar rozetini belirle
    decision_map = {
        "CR": ("TAM YANIT (Complete Response)", "Tüm hedef lezyonlar tamamen kaybolmuştur."),
        "PR": ("KISMİ YANIT (Partial Response)", "Hedef lezyonlarda anlamlı küçülme gözlenmiştir."),
        "SD": ("STABİL HASTALIK (Stable Disease)", "Hedef lezyonlarda belirgin değişiklik saptanmamıştır."),
        "PD": ("İLERLEYEN HASTALIK (Progressive Disease)", "Tümör yükünde artış ve/veya yeni lezyon tespit edilmiştir.")
    }
    
    decision_title, decision_desc = decision_map.get(decision, ("BELİRSİZ", "Değerlendirme yapılamamıştır."))
    
    # Lezyon detay tablosu
    lesion_details = ""
    if matched_pairs:
        lesion_details += "\n  Eşleşen Lezyonlar:\n"
        for idx, pair in enumerate(matched_pairs, 1):
            bl = pair.get("baseline_lesion", {})
            fu = pair.get("followup_lesion", {})
            d_change = pair.get("diameter_change_mm", 0)
            d_pct = pair.get("diameter_change_pct", 0)
            lesion_details += (
                f"    Lezyon #{idx}: "
                f"Çap {bl.get('diameter_mm', 0):.1f}mm → {fu.get('diameter_mm', 0):.1f}mm "
                f"({'+' if d_change >= 0 else ''}{d_change:.1f}mm, {'+' if d_pct >= 0 else ''}{d_pct:.1f}%)\n"
            )
    
    if new_lesions:
        lesion_details += f"\n  Yeni Lezyonlar: {len(new_lesions)} adet tespit edildi.\n"
        
    if disappeared:
        lesion_details += f"\n  Kaybolan/Regrese Lezyonlar: {len(disappeared)} adet.\n"
    
    # Radiomics sınıflandırma özeti
    radiomics_summary = ""
    if regions:
        radiomics_summary = "\nRADYOMİK SINIFLANDIRMA SONUÇLARI:\n"
        for r in regions:
            review_flag = " ⚠️ RADYOLOG ONAYI GEREKLİ" if r.get("needs_review") else ""
            radiomics_summary += (
                f"  Bölge #{r.get('region_id', '?')}: "
                f"{r.get('classification', 'Belirsiz')} "
                f"(Güven: %{r.get('confidence_score', 0) * 100:.0f}) "
                f"— HU Ort: {r.get('hu_mean', 0):.1f}{review_flag}\n"
            )
    
    mock_notice = ""
    if is_mock:
        mock_notice = (
            "\n╔════════════════════════════════════════════════════════╗\n"
            "║  ⚠️  DİKKAT: Bu rapor MOCK (simülasyon) verileriyle  ║\n"
            "║  üretilmiştir. Gerçek AI modeli henüz eğitilmemiştir. ║\n"
            "╚════════════════════════════════════════════════════════╝\n"
        )
    
    review_notice = ""
    if needs_review:
        review_notice = (
            "\n┌──────────────────────────────────────────────────────────┐\n"
            "│  🔍 RADYOLOG İNCELEMESİ GEREKLİ                        │\n"
            "│  Düşük güvenli (<%75) sınıflandırma bölgeleri mevcut.   │\n"
            "│  Bu bölgelerin radyolog tarafından doğrulanması gerekir.│\n"
            "└──────────────────────────────────────────────────────────┘\n"
        )
    
    report = f"""
════════════════════════════════════════════════════════════════
   ONKOLOJİK BT TAKİP — RECIST 1.1 KARAR DESTEK RAPORU
════════════════════════════════════════════════════════════════
{mock_notice}
HASTA BİLGİLERİ:
  Hasta ID     : {patient_info.get('patient_id', 'Bilinmiyor')}
  Hasta Adı    : {patient_info.get('patient_name', 'Bilinmiyor')}
  Çekim Tarihi : {patient_info.get('study_date', 'Bilinmiyor')}
  Rapor Tarihi : {datetime.now().strftime('%Y-%m-%d %H:%M')}

────────────────────────────────────────────────────────────────
KLİNİK DEĞERLENDİRME
────────────────────────────────────────────────────────────────

  RECIST 1.1 KARARI: {decision} — {decision_title}
  
  {decision_desc}

ÖLÇÜM DETAYLARI:
  Baz (t0) Hedef Lezyon Çap Toplamı (SOD) : {sod_baseline:.1f} mm
  Kontrol (t1) Hedef Lezyon Çap Toplamı    : {sod_followup:.1f} mm
  Tümör Yükü Değişimi                      : {'+' if change_pct >= 0 else ''}{change_pct:.1f}%
  Yeni Lezyon Varlığı                      : {'EVET' if new_lesion else 'HAYIR'}

LEZYON TAKİP DETAYLARI:
  Baseline (t0) Lezyon Sayısı  : {lesion_count_bl}
  Follow-up (t1) Lezyon Sayısı : {lesion_count_fu}
{lesion_details}
{radiomics_summary}
{review_notice}
────────────────────────────────────────────────────────────────
AÇIKLAMA: {explanation}
────────────────────────────────────────────────────────────────

SİSTEM NOTU: Bu rapor tamamen deterministik kural motoru (RECIST 1.1,
Eisenhauer et al., 2009) tarafından üretilmiş yapılandırılmış verilerin
standart radyoloji raporu diline dönüştürülmesiyle oluşturulmuştur.
LLM / yapay zeka tıbbi karar sürecinde kullanılmamıştır.

════════════════════════════════════════════════════════════════
"""
    return report.strip()


def _generate_llm_report(decision, explanation, change_pct, sod_baseline, sod_followup,
                          new_lesion, lesion_count_bl, lesion_count_fu,
                          matched_pairs, new_lesions, disappeared,
                          regions, needs_review, patient_info, api_key) -> str:
    """
    Gemini API ile doğrulanmış sayısal verileri akıcı klinik rapor diline çevirir.
    
    ÖNEMLİ: LLM'e GÖNDERİLEN prompt'ta açıkça belirtilir:
    - Kararı DEĞİŞTİRME
    - Yorum EKLEME  
    - Sadece verilen sayıları standart radyoloji diline ÇEVİR
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Lezyon eşleşme özetini hazırla
        lesion_summary = ""
        if matched_pairs:
            for idx, pair in enumerate(matched_pairs, 1):
                bl = pair.get("baseline_lesion", {})
                fu = pair.get("followup_lesion", {})
                lesion_summary += (
                    f"Lezyon #{idx}: Çap {bl.get('diameter_mm', 0):.1f}mm → "
                    f"{fu.get('diameter_mm', 0):.1f}mm "
                    f"(değişim: {pair.get('diameter_change_mm', 0):+.1f}mm)\n"
                )
        
        prompt = f"""Sen bir radyoloji rapor yazarısın. Aşağıdaki DOĞRULANMIŞ klinik verileri 
standart onkolojik BT takip raporu diline çevir.

KESİNLİKLE UYULMASI GEREKEN KURALLAR:
1. RECIST kararını DEĞİŞTİRME — bu karar deterministik kural motoru tarafından üretilmiştir.
2. Kendi yorumunu veya ek tanı EKLEME.
3. Sadece verilen sayıları ve kararı akıcı, profesyonel radyoloji diline ÇEVİR.
4. Raporu hem Türkçe hem İngilizce olarak yaz.

HASTA BİLGİLERİ:
- Hasta ID: {patient_info.get('patient_id', 'Bilinmiyor')}
- Hasta Adı: {patient_info.get('patient_name', 'Bilinmiyor')}
- Çekim Tarihi: {patient_info.get('study_date', 'Bilinmiyor')}

KLİNİK VERİLER:
- Baz (t0) SOD: {sod_baseline:.1f} mm
- Kontrol (t1) SOD: {sod_followup:.1f} mm
- Değişim: {change_pct:+.1f}%
- Yeni Lezyon: {'Var' if new_lesion else 'Yok'}
- Baseline Lezyon Sayısı: {lesion_count_bl}
- Follow-up Lezyon Sayısı: {lesion_count_fu}
- Yeni Tespit Edilen Lezyon: {len(new_lesions)} adet
- Kaybolan Lezyon: {len(disappeared)} adet

LEZYON DETAYLARI:
{lesion_summary if lesion_summary else 'Detay mevcut değil.'}

RECIST 1.1 KARARI: {decision}
AÇIKLAMA: {explanation}

Raporu standart radyoloji formatında, profesyonel ve kısa tut.
"""
        
        response = model.generate_content(prompt)
        
        llm_report = response.text
        
        # Güvenlik notu ekle
        llm_report += (
            "\n\n─────────────────────────────────────────────\n"
            "SİSTEM NOTU: Bu rapor, deterministik RECIST 1.1 kural motoru\n"
            "tarafından üretilmiş doğrulanmış verilerin Gemini API ile\n"
            "standart radyoloji diline çevrilmesiyle oluşturulmuştur.\n"
            "LLM tıbbi karar sürecinde kullanılmamıştır.\n"
            "─────────────────────────────────────────────"
        )
        
        return llm_report
        
    except ImportError:
        print("[WARNING] google-generativeai paketi yüklü değil. Şablon moduna düşülüyor...")
        return _generate_template_report(
            decision, explanation, change_pct, sod_baseline, sod_followup,
            new_lesion, lesion_count_bl, lesion_count_fu,
            matched_pairs, new_lesions, disappeared,
            regions, needs_review, patient_info=patient_info, is_mock=False
        )
    except Exception as e:
        print(f"[WARNING] LLM API hatası: {str(e)}. Şablon moduna düşülüyor...")
        return _generate_template_report(
            decision, explanation, change_pct, sod_baseline, sod_followup,
            new_lesion, lesion_count_bl, lesion_count_fu,
            matched_pairs, new_lesions, disappeared,
            regions, needs_review, patient_info=patient_info, is_mock=False
        )


if __name__ == "__main__":
    print("=== Klinik Rapor Üretici Modülü (report_generator.py) Yüklendi ===")
