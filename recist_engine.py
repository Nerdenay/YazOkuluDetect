"""
recist_engine.py
================
HepaRECIST-AI - Tek RECIST 1.1 karar motoru (Eisenhauer et al., 2009).

main.py (/recist-decision) ve pipeline.py aynı fonksiyonu çağırır; böylece iki
farklı kural seti oluşmaz.

Kurallar (hedef lezyonlar):
  CR : tüm hedef lezyonlar kayboldu (SOD = 0)
  PR : bazale göre SOD'de >= %30 azalma
  PD : nadire göre SOD'de >= %20 VE mutlak >= 5 mm artış
  SD : diğer
  NE : bazalda ölçülebilir hedef lezyon yok -> hedef yanıt değerlendirilemez

Yeni lezyon:
  - new_lesion_confirmed=True (radyolog onayı) -> PD
  - Sadece aday yeni lezyon(lar) varsa karar otomatik PD YAPILMAZ; karar hedef
    lezyonlardan verilir, 'provisional=True' ve 'decision_if_new_lesion_confirmed="PD"'
    döner. Rapor bunu açıkça belirtir.

Not: Lenf nodu kısa eksen kuralları (CR için tüm nodlar < 10 mm) ve hedef dışı
lezyonların kesin progresyonu bu sürümde otomatik değerlendirilmez.
"""

from typing import Optional, Dict, Any


def evaluate_recist(sod_baseline: float,
                    sod_followup: float,
                    sod_nadir: Optional[float] = None,
                    new_lesion_confirmed: bool = False,
                    new_lesion_candidates: int = 0) -> Dict[str, Any]:
    sod_baseline = float(sod_baseline or 0.0)
    sod_followup = float(sod_followup or 0.0)

    # Nadir: önceki ölçümlerin en küçüğü (bazal dahil). Takip değerinin kendisi nadir sayılmaz.
    if sod_nadir is None:
        nadir = sod_baseline
    else:
        nadir = min(float(sod_nadir), sod_baseline) if sod_baseline > 0 else float(sod_nadir)

    change_bl = ((sod_followup - sod_baseline) / sod_baseline * 100.0) if sod_baseline > 0 else 0.0
    abs_from_nadir = sod_followup - nadir
    if nadir > 0:
        change_nadir = abs_from_nadir / nadir * 100.0
    else:
        # Nadir 0 (önceden CR) iken herhangi bir yeniden ortaya çıkış: %20 kuralı
        # tanımsızdır; mutlak 5 mm kuralı uygulanır.
        change_nadir = float("inf") if sod_followup > 0 else 0.0

    # --- Hedef lezyon yanıtı ---
    if sod_baseline <= 0:
        target_decision = "NE"
        target_reason = ("Bazal tetkikte ölçülebilir hedef lezyon saptanmadığından "
                         "hedef lezyon yanıtı değerlendirilemez (NE).")
    elif change_nadir >= 20.0 and abs_from_nadir >= 5.0:
        target_decision = "PD"
        target_reason = (f"Hedef lezyon çap toplamı nadire ({nadir:.1f} mm) göre "
                         f"+{abs_from_nadir:.1f} mm"
                         + (f" (%{change_nadir:.1f})" if nadir > 0 else "")
                         + " artmıştır (PD).")
    elif sod_followup == 0:
        target_decision = "CR"
        target_reason = "Tüm hedef lezyonlar kaybolmuştur (CR). Lenf nodu kısa eksenleri ayrıca doğrulanmalıdır."
    elif change_bl <= -30.0:
        target_decision = "PR"
        target_reason = (f"Hedef lezyon çap toplamı bazale ({sod_baseline:.1f} mm) göre "
                         f"%{abs(change_bl):.1f} azalmıştır (PR).")
    else:
        target_decision = "SD"
        target_reason = (f"PR veya PD kriteri karşılanmamıştır (SD). "
                         f"Bazale göre değişim: %{change_bl:+.1f}.")

    # --- Yeni lezyon ---
    provisional = False
    if new_lesion_confirmed:
        decision = "PD"
        reason = "Radyolog tarafından onaylanmış yeni lezyon mevcuttur (PD). " + target_reason
    else:
        decision = target_decision
        reason = target_reason
        if new_lesion_candidates > 0 and target_decision != "PD":
            provisional = True
            reason += (f" DİKKAT: {new_lesion_candidates} aday yeni lezyon radyolog onayı "
                       f"beklemektedir; onaylanırsa karar PD olur.")

    return {
        "decision": decision,
        "target_response": target_decision,
        "explanation": reason,
        "provisional": provisional,
        "decision_if_new_lesion_confirmed": "PD" if provisional else decision,
        "baseline_sod": round(sod_baseline, 2),
        "followup_sod": round(sod_followup, 2),
        "nadir_sod": round(nadir, 2),
        "change_from_baseline_pct": round(change_bl, 2),
        "change_from_nadir_pct": round(change_nadir, 2) if change_nadir != float("inf") else None,
        # Geriye dönük uyumluluk (C# arayüzü ve rapor modülü)
        "change_percentage": round(change_bl, 2),
        "new_lesion": bool(new_lesion_confirmed),
        "new_lesion_candidates": int(new_lesion_candidates),
    }
