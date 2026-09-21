"""
LeadLens — Lead Scorer
Computes a composite lead priority score (0-100) and assigns a tier:
  Hot (≥70) | Warm (40-69) | Cold (<40)

Scoring formula:
  score = intent_score (40) + entity_richness (30) + language_clarity (15) + specificity (15)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Tuple


def compute_score(
    intent: str,
    confidence: float,
    entities: Dict,
    lang: str,
    lang_conf: float,
) -> Tuple[int, str, Dict[str, float]]:
    """
    Compute lead priority score.

    Args:
        intent      : classified intent string
        confidence  : intent classification confidence (0-1)
        entities    : dict from entity_extractor.extract_entities()
        lang        : detected language code
        lang_conf   : language detection confidence (0-1)

    Returns:
        (score, priority, breakdown)
        score     : int 0-100
        priority  : 'Hot' | 'Warm' | 'Cold'
        breakdown : dict showing component scores
    """
    from config.settings import (
        INTENT_WEIGHTS, SCORE_WEIGHTS,
        PRIORITY_HOT_THRESHOLD, PRIORITY_WARM_THRESHOLD,
    )

    # ── 1. Intent score (max 40 pts) ─────────────────────────────────────────
    intent_weight  = INTENT_WEIGHTS.get(intent, 0.0)
    # Scale by confidence so uncertain predictions score lower
    intent_score   = intent_weight * confidence * SCORE_WEIGHTS["intent"]

    # ── 2. Entity richness (max 30 pts = 7.5 per entity) ─────────────────────
    entity_count   = sum(1 for v in entities.values() if v is not None)
    entity_score   = min(entity_count, 4) * (SCORE_WEIGHTS["entities"] / 4)

    # ── 3. Language clarity (max 15 pts) ─────────────────────────────────────
    # Reward high-confidence language detection (ambiguous text = lower score)
    lang_score = lang_conf * SCORE_WEIGHTS["language"]

    # ── 4. Specificity bonus (max 15 pts) ─────────────────────────────────────
    # Full bonus: quantity + budget both present
    # Partial: only one of them present
    has_qty    = entities.get("quantity") is not None
    has_budget = entities.get("budget") is not None
    has_product = entities.get("product") is not None

    if has_qty and has_budget:
        spec_score = SCORE_WEIGHTS["specificity"] * 1.00  # full 15 pts
    elif has_qty or has_budget:
        spec_score = SCORE_WEIGHTS["specificity"] * 0.60  # 9 pts
    elif has_product:
        spec_score = SCORE_WEIGHTS["specificity"] * 0.30  # 4.5 pts
    else:
        spec_score = 0.0

    # ── Total ─────────────────────────────────────────────────────────────────
    total = intent_score + entity_score + lang_score + spec_score
    score = max(0, min(100, round(total)))

    # Spam always scores 0
    if intent == "spam":
        score = 0

    # ── Priority tier ─────────────────────────────────────────────────────────
    if score >= PRIORITY_HOT_THRESHOLD:
        priority = "Hot"
    elif score >= PRIORITY_WARM_THRESHOLD:
        priority = "Warm"
    else:
        priority = "Cold"

    breakdown = {
        "intent_score":   round(intent_score, 1),
        "entity_score":   round(entity_score, 1),
        "language_score": round(lang_score, 1),
        "specificity":    round(spec_score, 1),
        "total":          score,
    }

    return (score, priority, breakdown)


def get_recommended_action(intent: str, priority: str, entities: Dict, lang: str = "en") -> str:
    """
    Generate a detailed, multilingual recommended action string for the sales team.
    The action is written in the same language as the customer's inquiry so it can
    be forwarded directly as a reply template.
    """
    if intent == "spam":
        spam_msgs = {
            "en":       "🚫 Mark as spam. Block sender and do not respond.",
            "hi":       "🚫 स्पैम के रूप में चिह्नित करें। प्रेषक को ब्लॉक करें और जवाब न दें।",
            "mr":       "🚫 स्पॅम म्हणून चिन्हांकित करा. प्रेषकाला ब्लॉक करा आणि उत्तर देऊ नका.",
            "mr_roman": "🚫 Spam mhanun chinhankeet kara. Presakala block kara, uttar deu naka.",
            "hinglish": "🚫 Spam mark karo. Sender ko block karo, reply mat karo.",
        }
        return spam_msgs.get(lang, spam_msgs["en"])

    product = entities.get("product")
    qty     = entities.get("quantity")
    budget  = entities.get("budget")
    loc     = entities.get("location")

    # ── Urgency prefix based on priority ──────────────────────────────────────
    urgency = {
        "Hot": {
            "en":       "🔴 URGENT — Call the customer within 1 hour.",
            "hi":       "🔴 अत्यंत जरूरी — 1 घंटे के अंदर ग्राहक को कॉल करें।",
            "mr":       "🔴 अत्यंत तातडीचे — 1 तासात ग्राहकाला कॉल करा.",
            "mr_roman": "🔴 Atyanth tatadiche — 1 tasat grahakaala call kara.",
            "hinglish": "🔴 URGENT — 1 ghante ke andar customer ko call karo.",
        },
        "Warm": {
            "en":       "🟡 Follow up within 24 hours.",
            "hi":       "🟡 24 घंटे के अंदर फॉलो अप करें।",
            "mr":       "🟡 24 तासांत फॉलो अप करा.",
            "mr_roman": "🟡 24 tasant follow up kara.",
            "hinglish": "🟡 24 ghante ke andar follow up karo.",
        },
        "Cold": {
            "en":       "🔵 Add to nurture sequence.",
            "hi":       "🔵 नर्चर सीक्वेंस में जोड़ें।",
            "mr":       "🔵 Nurture sequence मध्ये जोडा.",
            "mr_roman": "🔵 Nurture sequence madhe joda.",
            "hinglish": "🔵 Nurture sequence mein add karo.",
        },
    }

    # ── Intent-specific action templates ──────────────────────────────────────
    if intent == "bulk_order":
        actions = {
            "en": (
                f"Send the bulk pricing catalogue immediately."
                + (f" Confirm stock availability for {qty['raw']}." if qty else "")
                + (f" Arrange delivery logistics to {loc}." if loc else "")
                + (f" Customer's budget is {budget['display']} — prepare a matching quote." if budget else "")
                + (f" Highlight {product} options with volume discounts." if product else "")
                + " Ask for a site visit or video call to close faster."
            ),
            "hi": (
                f"थोक मूल्य सूची तुरंत भेजें।"
                + (f" {qty['raw']} की उपलब्धता की पुष्टि करें।" if qty else "")
                + (f" {loc} तक डिलीवरी की व्यवस्था करें।" if loc else "")
                + (f" ग्राहक का बजट {budget['display']} है — उसके अनुसार कोटेशन तैयार करें।" if budget else "")
                + (f" {product} के विकल्प और बल्क डिस्काउंट बताएं।" if product else "")
                + " डील जल्दी बंद करने के लिए साइट विजिट या वीडियो कॉल का सुझाव दें।"
            ),
            "mr": (
                f"घाऊक किंमत यादी त्वरित पाठवा."
                + (f" {qty['raw']} साठी स्टॉकची उपलब्धता निश्चित करा." if qty else "")
                + (f" {loc} पर्यंत डिलिव्हरीची व्यवस्था करा." if loc else "")
                + (f" ग्राहकाचे बजेट {budget['display']} आहे — त्यानुसार कोटेशन तयार करा." if budget else "")
                + (f" {product} चे पर्याय आणि बल्क सूट सांगा." if product else "")
                + " व्यवहार लवकर बंद करण्यासाठी साइट व्हिजिट किंवा व्हिडिओ कॉल सुचवा."
            ),
            "mr_roman": (
                f"Ghau kimmat yadi tatkal pathva."
                + (f" {qty['raw']} sathi stock uplabdhata nishchit kara." if qty else "")
                + (f" {loc} paryant delivery vyavastha kara." if loc else "")
                + (f" Grahakacha budget {budget['display']} aahe — tyanusar quotation tayar kara." if budget else "")
                + (f" {product} che paryay ani bulk sut sanga." if product else "")
                + " Deal lavkar band karnyasathi site visit kiva video call suchva."
            ),
            "hinglish": (
                f"Bulk pricing catalogue turant bhejo."
                + (f" {qty['raw']} ki availability confirm karo." if qty else "")
                + (f" {loc} tak delivery arrange karo." if loc else "")
                + (f" Customer ka budget {budget['display']} hai — ussi ke hisaab se quote banao." if budget else "")
                + (f" {product} ke options aur volume discounts highlight karo." if product else "")
                + " Deal jaldi close karne ke liye site visit ya video call suggest karo."
            ),
        }

    elif intent == "price_query":
        actions = {
            "en": (
                f"Share the updated product price list."
                + (f" Focus on {product} — prepare a specific quote." if product else "")
                + (f" Customer's budget is around {budget['display']} — shortlist options in this range." if budget else "")
                + (f" Confirm whether delivery to {loc} is available and at what cost." if loc else "")
                + " Offer a free consultation call to understand their exact requirement."
            ),
            "hi": (
                f"अपडेटेड प्राइस लिस्ट शेयर करें।"
                + (f" {product} पर ध्यान दें — विशेष कोटेशन तैयार करें।" if product else "")
                + (f" ग्राहक का बजट लगभग {budget['display']} है — उसी रेंज के विकल्प सुझाएं।" if budget else "")
                + (f" {loc} तक डिलीवरी उपलब्ध है या नहीं और लागत क्या होगी, यह बताएं।" if loc else "")
                + " उनकी सटीक जरूरत समझने के लिए फ्री कंसल्टेशन कॉल का प्रस्ताव दें।"
            ),
            "mr": (
                f"अपडेटेड किंमत यादी शेअर करा."
                + (f" {product} वर लक्ष द्या — विशेष कोटेशन तयार करा." if product else "")
                + (f" ग्राहकाचे बजेट सुमारे {budget['display']} आहे — त्याच रेंजमधील पर्याय सुचवा." if budget else "")
                + (f" {loc} पर्यंत डिलिव्हरी उपलब्ध आहे का आणि खर्च किती, ते सांगा." if loc else "")
                + " त्यांची नेमकी गरज समजून घेण्यासाठी मोफत सल्ला कॉलचा प्रस्ताव द्या."
            ),
            "mr_roman": (
                f"Updated kimmat yadi share kara."
                + (f" {product} var laks dya — vishesh quotation tayar kara." if product else "")
                + (f" Grahakacha budget sadharan {budget['display']} aahe — tyach range madhil paryay suchva." if budget else "")
                + (f" {loc} paryant delivery uplabdh aahe ka ani kharch kiti, te sanga." if loc else "")
                + " Tyanchya nemakya garajanusara free consultation call cha prastaav dya."
            ),
            "hinglish": (
                f"Updated price list share karo."
                + (f" {product} pe focus karo — specific quote banao." if product else "")
                + (f" Customer ka budget lagbhag {budget['display']} hai — usi range ke options suggest karo." if budget else "")
                + (f" {loc} tak delivery available hai ya nahi aur cost kya hogi, batao." if loc else "")
                + " Unki exact zaroorat samajhne ke liye free consultation call offer karo."
            ),
        }

    elif intent == "complaint":
        actions = {
            "en": (
                "Escalate to the support team immediately."
                + (f" Issue involves: {product}." if product else "")
                + (f" Customer is in {loc} — check if a field visit is needed." if loc else "")
                + " Request the order ID and date of purchase."
                + " Check if the item is under warranty and initiate replacement/refund process."
                + " Send an apology message within 2 hours to retain customer trust."
            ),
            "hi": (
                "सपोर्ट टीम को तुरंत सूचित करें।"
                + (f" समस्या: {product} से संबंधित है।" if product else "")
                + (f" ग्राहक {loc} में हैं — फील्ड विजिट की जरूरत है या नहीं, जांचें।" if loc else "")
                + " ऑर्डर ID और खरीद की तारीख मांगें।"
                + " वारंटी की जांच करें और रिप्लेसमेंट/रिफंड प्रक्रिया शुरू करें।"
                + " ग्राहक का विश्वास बनाए रखने के लिए 2 घंटे के अंदर माफ़ी का संदेश भेजें।"
            ),
            "mr": (
                "सपोर्ट टीमला ताबडतोब कळवा."
                + (f" समस्या: {product} संबंधित आहे." if product else "")
                + (f" ग्राहक {loc} मध्ये आहेत — फील्ड व्हिजिटची गरज आहे का ते तपासा." if loc else "")
                + " ऑर्डर ID आणि खरेदीची तारीख मागवा."
                + " वॉरंटी तपासा आणि बदली/परतावा प्रक्रिया सुरू करा."
                + " ग्राहकाचा विश्वास टिकवण्यासाठी 2 तासांत माफीचा संदेश पाठवा."
            ),
            "mr_roman": (
                "Support teamla tatkal kalava."
                + (f" Samasya: {product} sambandhit aahe." if product else "")
                + (f" Grahak {loc} madhe aahet — field visit chi garaj aahe ka te tapasa." if loc else "")
                + " Order ID ani kharedichi tarikh magva."
                + " Warranty tapasa ani badali/paratava prakriya suru kara."
                + " Grahakache vishwas tikvanyasathi 2 tasant maficha sandesh pathva."
            ),
            "hinglish": (
                "Support team ko turant inform karo."
                + (f" Issue {product} se related hai." if product else "")
                + (f" Customer {loc} mein hai — field visit ki zaroorat hai ya nahi, check karo." if loc else "")
                + " Order ID aur purchase date maango."
                + " Warranty check karo aur replacement/refund process start karo."
                + " Customer ka trust banaye rakhne ke liye 2 ghante mein maafi ka message bhejo."
            ),
        }
    else:
        actions = {
            "en": "Review this inquiry manually and assign to the appropriate team.",
            "hi": "इस इनक्वायरी की समीक्षा करें और उचित टीम को असाइन करें।",
            "mr": "ही इन्क्वायरी तपासा आणि योग्य टीमला असाइन करा.",
            "mr_roman": "Hi inquiry tapasa ani yogya teamla assign kara.",
            "hinglish": "Is inquiry ko manually review karo aur sahi team ko assign karo.",
        }

    # ── Combine urgency prefix + action ───────────────────────────────────────
    lang_key = lang if lang in ("en", "hi", "mr", "mr_roman", "hinglish") else "en"
    u = urgency.get(priority, urgency["Cold"]).get(lang_key, urgency["Cold"]["en"])
    a = actions.get(lang_key, actions["en"]) if isinstance(actions, dict) else actions
    return f"{u} {a}"
