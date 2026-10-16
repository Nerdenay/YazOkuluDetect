"""
report_generator.py
===================
HepaRECIST-AI - Faz 5: Klinik Radyoloji Rapor Üretim Motoru
- Düzce Üniversitesi Tıp Fakültesi Araştırma Hastanesi Resmi Rapor Şablonu
- Deterministik RECIST 1.1 Kararı ve Radyomik Doğrulama (Label 8, Sphericity Ψ)
- LLM (Gemini) entegrasyonu VARSAYILAN OLARAK KAPALI. Açılabilmesi için hem
  use_llm=True hem de ortam değişkeni HEPARECIST_ALLOW_EXTERNAL_LLM=1 gerekir.
  Dış servise hasta adı, ID veya tarih GÖNDERİLMEZ; yalnızca anonim ölçümler gider.
  Hasta bilgileri rapora LLM çıktısından sonra yerel olarak eklenir. (KVKK)
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


def generate_clinical_report(pipeline_results: Dict[str, Any],
                              patient_info: Optional[Dict[str, str]] = None,
                              use_llm: bool = False,
                              llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Pipeline çıktılarını resmi onkolojik radyoloji raporuna dönüştürür.
    """
    if patient_info is None:
        patient_info = {
            "patient_id": "Bilinmiyor",
            "patient_name": "Bilinmiyor",
            "study_date": datetime.now().strftime("%Y-%m-%d")
        }

    # Pipeline sonuçlarını güvenli ve esnek bir şekilde ayrıştır
    recist = pipeline_results.get("recist_decision", {})
    segmentation = pipeline_results.get("segmentation", {})
    radiomics = pipeline_results.get("radiomics", {})
    longitudinal = pipeline_results.get("longitudinal", {})
    matching = longitudinal.get("matching", {})
    
    # recist_metrics veya recist_input desteği
    recist_data = longitudinal.get("recist_metrics", longitudinal.get("recist_input", {}))

    decision = recist.get("decision", "SD")
    explanation = recist.get("explanation", "")
    change_pct = recist.get("change_percentage", recist_data.get("sod_change_pct", 0.0))
    sod_baseline = recist.get("baseline_sod", recist_data.get("sod_baseline", 0.0))
    sod_followup = recist.get("followup_sod", recist_data.get("sod_followup", 0.0))
    new_lesion = recist.get("new_lesion", recist_data.get("has_new_lesions", False))

    lesion_count_bl = longitudinal.get("baseline_targets", longitudinal.get("baseline_lesion_count", 0))
    lesion_count_fu = longitudinal.get("followup_lesions", longitudinal.get("followup_lesion_count", 0))
    
    matched_pairs = matching.get("matched_pairs", [])
    new_lesions = matching.get("new_lesions", [])
    disappeared = matching.get("disappeared_lesions", [])

    regions = radiomics.get("regions", [])
    needs_review = radiomics.get("requires_radiologist_review", False)
    is_mock = segmentation.get("is_mock", False)
    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    review = _collect_review_items(longitudinal, recist)
    llm_allowed = os.environ.get("HEPARECIST_ALLOW_EXTERNAL_LLM", "0") == "1"
    if use_llm and not llm_allowed:
        print("[INFO] Dış LLM kullanımı ortam değişkeniyle izin verilmediği için kapalı "
              "(HEPARECIST_ALLOW_EXTERNAL_LLM=1). Şablon rapor kullanılıyor.")

    # LLM veya Şablon Seçimi
    if use_llm and llm_api_key and llm_allowed:
        report_text = _generate_llm_report(
            decision=decision, explanation=explanation, change_pct=change_pct,
            sod_baseline=sod_baseline, sod_followup=sod_followup,
            new_lesion=new_lesion, lesion_count_bl=lesion_count_bl,
            lesion_count_fu=lesion_count_fu, matched_pairs=matched_pairs,
            new_lesions=new_lesions, disappeared=disappeared,
            regions=regions, needs_review=needs_review,
            patient_info=patient_info, api_key=llm_api_key, review=review
        )
        report_source = "LLM Destekli (anonim veriyle)"
    else:
        report_text = _generate_template_report(
            decision=decision, explanation=explanation, change_pct=change_pct,
            sod_baseline=sod_baseline, sod_followup=sod_followup,
            new_lesion=new_lesion, lesion_count_bl=lesion_count_bl,
            lesion_count_fu=lesion_count_fu, matched_pairs=matched_pairs,
            new_lesions=new_lesions, disappeared=disappeared,
            regions=regions, needs_review=needs_review,
            patient_info=patient_info, is_mock=is_mock, review=review
        )
        report_source = "Düzce Üniversitesi Resmi Şablonu"

    return {
        "status": "success",
        "report_text": report_text,
        "report_source": report_source,
        "generated_at": report_timestamp,
        "recist_decision": decision,
        "requires_radiologist_review": bool(needs_review or review["any"]),
        "is_ai_mock": is_mock
    }


def _collect_review_items(longitudinal: Dict[str, Any], recist: Dict[str, Any]) -> Dict[str, Any]:
    """Radyolog onayı gerektiren maddeleri toplar."""
    matching = longitudinal.get("matching", {})
    qc = longitudinal.get("registration_qc", {})
    unmatched = matching.get("unmatched_followup_review", [])
    items = {
        "provisional": bool(recist.get("provisional", False)),
        "decision_if_confirmed": recist.get("decision_if_new_lesion_confirmed"),
        "candidates": [u for u in unmatched if u.get("status") == "candidate_new"],
        "missed_at_baseline": [u for u in unmatched if u.get("status") in
                               ("possibly_missed_at_baseline", "present_at_baseline_mask")],
        "too_small": [u for u in unmatched if u.get("status") == "too_small"],
        "qc": qc,
        "targets": longitudinal.get("target_lesions", []),
    }
    items["any"] = bool(items["candidates"] or items["missed_at_baseline"] or
                        matching.get("disappeared_lesions") or
                        (qc and qc.get("reliable") is not True))
    return items


def _review_section(review: Optional[Dict[str, Any]], disappeared) -> str:
    if not review:
        return "  • Ek onay maddesi yok."
    lines = []
    qc = review.get("qc") or {}
    if qc:
        dice = ", ".join(f"{k}: {v:.2f}" for k, v in qc.get("organ_dice", {}).items()) or "ölçülemedi"
        lines.append(f"  • Kayıt kalitesi (organ Dice): {dice} — {qc.get('message', '')}")
    for c in review.get("candidates", []):
        fl = c["followup_lesion"]
        lines.append(f"  • [ONAY] Aday yeni lezyon #{fl['id']} ({fl.get('organ','?')}, {fl['diameter_mm']:.1f} mm): {c.get('reason','')}")
    for c in review.get("missed_at_baseline", []):
        fl = c["followup_lesion"]
        lines.append(f"  • [ONAY] Takip lezyonu #{fl['id']} ({fl['diameter_mm']:.1f} mm): {c.get('reason','')}")
    for d in disappeared or []:
        bl = d.get("baseline_lesion", {})
        lines.append(f"  • [ONAY] Bazal lezyon #{bl.get('id','?')} ({bl.get('diameter_mm',0):.1f} mm) takipte bulunamadı.")
    if review.get("too_small"):
        lines.append(f"  • Boyut eşiği altında {len(review['too_small'])} küçük takip odağı (karara dahil edilmedi).")
    if review.get("provisional"):
        lines.append(f"  • KARAR GEÇİCİDİR: Aday yeni lezyon onaylanırsa kategori "
                     f"{review.get('decision_if_confirmed', 'PD')} olur.")
    return "\n".join(lines) if lines else "  • Ek onay maddesi yok."


def _generate_template_report(decision, explanation, change_pct, sod_baseline, sod_followup,
                                new_lesion, lesion_count_bl, lesion_count_fu,
                                matched_pairs, new_lesions, disappeared,
                                regions, needs_review, patient_info, is_mock, review=None) -> str:
    """
    Düzce Üniversitesi Tıp Fakültesi Onkolojik Radyoloji resmi şablonu.
    """
    decision_titles = {
        "CR": "TAM YANIT (Complete Response - CR)",
        "PR": "KISMİ YANIT (Partial Response - PR)",
        "SD": "STABİL HASTALIK (Stable Disease - SD)",
        "PD": "İLERLEYEN HASTALIK (Progressive Disease - PD)",
        "NE": "DEĞERLENDİRİLEMEZ (Not Evaluable - NE)"
    }
    decision_full = decision_titles.get(decision, f"DEĞERLENDİRİLEMEDİ ({decision})")

    # 1. Eşleşen Lezyonlar Tablosu (Hem flat hem nested anahtar desteği)
    lesion_lines = []
    targets = (review or {}).get("targets") or []
    if targets:
        for idx, t in enumerate(targets, 1):
            bl_d, fu_d = t["baseline_diameter"], t["followup_diameter"]
            d_ch = fu_d - bl_d
            d_pct = (d_ch / bl_d * 100) if bl_d > 0 else 0.0
            note = "" if t.get("status") == "matched" else "  [takipte bulunamadı — onay gerekli]"
            lesion_lines.append(
                f"  • Hedef Lezyon #{idx} ({t.get('organ','?')}): t0: {bl_d:.1f} mm ➔ t1: {fu_d:.1f} mm "
                f"({'+' if d_ch >= 0 else ''}{d_ch:.1f} mm, {'+' if d_pct >= 0 else ''}{d_pct:.1f}%){note}")
        lesion_details_str = "\n".join(lesion_lines)
    elif matched_pairs:
        for idx, pair in enumerate(matched_pairs, 1):
            bl_d = pair.get("baseline_diameter") or pair.get("baseline_lesion", {}).get("diameter_mm", 0.0)
            fu_d = pair.get("followup_diameter") or pair.get("followup_lesion", {}).get("diameter_mm", 0.0)
            d_ch = pair.get("diameter_change_mm", fu_d - bl_d)
            d_pct = pair.get("diameter_change_pct", ((fu_d - bl_d)/bl_d * 100) if bl_d > 0 else 0.0)
            
            lesion_lines.append(
                f"  • Hedef Lezyon #{idx}: t0: {bl_d:.1f} mm ➔ t1: {fu_d:.1f} mm "
                f"({'+' if d_ch >= 0 else ''}{d_ch:.1f} mm, {'+' if d_pct >= 0 else ''}{d_pct:.1f}%)"
            )
        lesion_details_str = "\n".join(lesion_lines)
    else:
        lesion_details_str = "  • Eşleşen hedef lezyon bulunmamaktadır."

    # 2. Yeni ve Kaybolan Lezyonlar
    if new_lesion:
        new_lesion_str = "  • Radyolog onaylı yeni lezyon mevcuttur (PD kriteri)."
    elif new_lesions:
        new_lesion_str = f"  • Aday yeni lezyon: {len(new_lesions)} adet (radyolog onayı bekliyor; henüz PD sayılmadı)"
    else:
        new_lesion_str = "  • Aday yeni lezyon saptanmamıştır."
    disappeared_str = f"  • Takipte bulunamayan bazal lezyon: {len(disappeared)} adet (doğrulama gerekli)" if disappeared else ""
    review_str = _review_section(review, disappeared)

    # 3. Radyomik ve 3D Küresellik Analizi
    radiomics_lines = []
    if regions:
        for r in regions:
            rev_badge = " [⚠️ HEKİM ONAYI GEREKLİ]" if r.get("needs_review") else ""
            psi = r.get("sphericity_psi", 0.0)
            radiomics_lines.append(
                f"  • Odak #{r.get('region_id')}: {r.get('classification')} | "
                f"Yoğunluk: {r.get('hu_mean'):.1f} HU | 3D Küresellik (Ψ): {psi:.2f}{rev_badge}"
            )
        radiomics_str = "\n".join(radiomics_lines)
    else:
        radiomics_str = "  • Aday lezyon bölgesi yok."

    mock_badge = "\n*** DİKKAT: BU RAPOR AI TEST/MOCK VERİLERİYLE ÜRETİLMİŞTİR ***\n" if is_mock else ""

    report = f"""================================================================================
T.C. DÜZCE ÜNİVERSİTESİ TIP FAKÜLTESİ SAĞLIK UYGULAMA VE ARAŞTIRMA MERKEZİ
RADYOLOJİ ANABİLİM DALI — 4D ONKOLOJİK BT TAKİP VE RECIST 1.1 RAPORU
================================================================================{mock_badge}
HASTA VE TETKİK BİLGİLERİ:
  Protokol No / Hasta ID : {patient_info.get('patient_id', 'Bilinmiyor')}
  Hasta Adı Soyadı       : {patient_info.get('patient_name', 'Bilinmiyor')}
  Tetkik / Çekim Tarihi  : {patient_info.get('study_date', 'Bilinmiyor')}
  Rapor Tanzim Tarihi    : {datetime.now().strftime('%d.%m.%Y %H:%M')}
  İnceleme Alanı         : Batın BT (analiz 1.0 mm izotropik yeniden örneklenmiş seride)

KLİNİK ENDİKASYON VE YÖNTEM:
  Longitudinal kemoterapi / immünoterapi tedavi yanıtı takibi.
  Baseline (t0) ve Follow-up (t1) serileri SimpleITK Deformable B-Spline Registration
  ile eşleştirilmiş, hedef lezyonlar aksiyel 2D eksende RECIST 1.1 kriterlerine göre 
  hesaplanmıştır.

TÜMÖR YÜKÜ VE ÇAP ÖLÇÜMLERİ (RECIST 1.1):
  • Bazal (t0) Hedef Lezyon Çap Toplamı (SOD) : {sod_baseline:.1f} mm
  • Takip (t1) Hedef Lezyon Çap Toplamı (SOD) : {sod_followup:.1f} mm
  • Tümör Yükü Değişim Yüzdesi (Δ%)           : {'+' if change_pct >= 0 else ''}{change_pct:.1f} %
  • Yeni Lezyon (onaylı)                      : {'EVET' if new_lesion else 'HAYIR'}

HEDEF LEZYON TAKİP VE EŞLEŞTİRME AYRINTILARI:
{lesion_details_str}
{new_lesion_str}
{disappeared_str}

RADYOLOG ONAYI GEREKEN MADDELER VE KALİTE KONTROL:
{review_str}

DANSİTE ÖN-KATEGORİLERİ (tanı değildir, doğrulanmamış eşikler):
{radiomics_str}

SONUÇ VE TEDAVİ YANITI KARARI:
  ► RECIST 1.1 YANIT KATEGORİSİ : {decision_full}
  ► DEĞERLENDİRME VE GEREKÇE    : {explanation}

KLİNİK BİLGİ NOTU:
Bu rapor bir KARAR DESTEK taslağıdır; RECIST 1.1 (Eisenhauer et al., 2009) kuralları
otomatik ölçümlere uygulanmıştır. Lenf nodu kısa eksen kuralları ve hedef dışı lezyonlar
otomatik değerlendirilmemiştir. Rapor radyolog onayı olmadan klinik karar için kullanılamaz.
================================================================================
"""
    return report.strip()


def _generate_llm_report(decision, explanation, change_pct, sod_baseline, sod_followup,
                          new_lesion, lesion_count_bl, lesion_count_fu,
                          matched_pairs, new_lesions, disappeared,
                          regions, needs_review, patient_info, api_key, review=None) -> str:
    """
    Gemini 2.0 Flash ile Düzce Üniversitesi resmi rapor diline aktarır.
    Halüsinasyonu önlemek için temperature=0.0 ile çalışır.
    """
    try:
        # pyrefly: ignore [missing-import]
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={"temperature": 0.0}
        )

        lesion_lines = []
        for idx, pair in enumerate(matched_pairs, 1):
            bl_d = pair.get("baseline_diameter") or pair.get("baseline_lesion", {}).get("diameter_mm", 0.0)
            fu_d = pair.get("followup_diameter") or pair.get("followup_lesion", {}).get("diameter_mm", 0.0)
            lesion_lines.append(f"Hedef Lezyon #{idx}: t0: {bl_d:.1f}mm -> t1: {fu_d:.1f}mm")

        prompt = f"""Sen Düzce Üniversitesi Tıp Fakültesi Radyoloji Anabilim Dalı'nda uzman bir radyoloji rapor sekreterisin.
Aşağıda deterministik kural motoru tarafından üretilmiş KESİN VE DEĞİŞTİRİLEMEZ klinik ölçüm verileri verilmiştir.
Görevin: Bu verileri resmi, profesyonel bir onkolojik BT takip raporu haline getirmektir.

KESİN KURALLAR:
1. RECIST 1.1 KARARINI ASLA DEĞİŞTİRME: Karar '{decision}' olarak verilmiştir.
2. Sayısal değerleri (SOD, yüzde, milimetre) AYNEN koru, asla yuvarlama veya değiştirme.
3. Kendinden ek tanı, organ patolojisi veya metastaz uydurma (Halüsinasyon yasaktır).
4. Raporu Türkçe resmi hastane formatında yaz. Hasta kimlik bilgisi alanı YAZMA (yerel olarak eklenecek).
5. Aday yeni lezyonları kesin yeni lezyon gibi yazma; radyolog onayı beklediğini belirt.

SAYISAL VERİLER:
- Bazal SOD: {sod_baseline:.1f} mm
- Takip SOD: {sod_followup:.1f} mm
- Değişim: {change_pct:+.1f}%
- Onaylı Yeni Lezyon: {'Var' if new_lesion else 'Yok'}
- Aday Yeni Lezyon Sayısı (onay bekliyor): {len(new_lesions)}
- Takipte Bulunamayan Bazal Lezyon Sayısı: {len(disappeared)}
- Karar geçici mi: {'Evet' if (review or {}).get('provisional') else 'Hayır'}
- Lezyon Ölçümleri: {', '.join(lesion_lines) if lesion_lines else 'Yok'}

RECIST 1.1 KARARI: {decision}
GEREKÇE: {explanation}
"""

        response = model.generate_content(prompt)
        header = (f"Hasta ID: {patient_info.get('patient_id', 'Bilinmiyor')} | "
                  f"Hasta: {patient_info.get('patient_name', 'Bilinmiyor')} | "
                  f"Tarih: {patient_info.get('study_date', 'Bilinmiyor')}\n\n")
        llm_text = header + response.text + "\n\nONAY MADDELERİ:\n" + _review_section(review, disappeared)

        disclaimer = (
            "\n\n────────────────────────────────────────────────────────────────\n"
            "BİLGİ NOTU: Bu rapor Düzce Üniversitesi HepaRECIST-AI Karar Motoru verilerinin\n"
            "Gemini API ile resmi klinik dile çevrilmesiyle oluşturulmuştur.\n"
            "Yapay zeka modeli tıbbi karar vermemiş, sadece doğrulanmış verileri aktarmıştır.\n"
            "────────────────────────────────────────────────────────────────"
        )
        return llm_text + disclaimer

    except Exception as e:
        print(f"[WARNING] LLM çağrısı başarısız ({e}). Standart şablona dönülüyor...")
        return _generate_template_report(
            decision, explanation, change_pct, sod_baseline, sod_followup,
            new_lesion, lesion_count_bl, lesion_count_fu,
            matched_pairs, new_lesions, disappeared,
            regions, needs_review, patient_info=patient_info, is_mock=False, review=review
        )


if __name__ == "__main__":
    print("=== HepaRECIST-AI: Klinik Rapor Üretim Modülü (report_generator.py) ===")
