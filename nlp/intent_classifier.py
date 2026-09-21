"""
LeadLens — Intent Classifier
Classifies customer inquiries into:
  price_query | bulk_order | complaint | spam

Strategy:
  1. Fast rule-based classifier using multilingual keyword dictionaries.
  2. Optional XLM-RoBERTa zero-shot fallback for ambiguous cases
     (enabled via config.settings.USE_TRANSFORMER_FALLBACK).
"""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Tuple

# ── Multilingual Keyword Dictionaries ─────────────────────────────────────────
# Each dict maps intent → set of trigger phrases/tokens
# Patterns are matched as whole-word or substring depending on type.

_BULK_ORDER_PATTERNS = {
    # English
    r"\bbulk\b", r"\bwholesale\b", r"\blarge\s+(?:order|quantity|qty)\b",
    r"\b\d+\s*(?:kg|kgs|ton|tons|tonne|tonnes|quintal|quintals)\b",
    r"\b\d+\s*(?:pieces?|pcs|units?|nos?|dozen|dozens|gross)\b",
    r"\b(?:500|1000|2000|5000|10000)\s*(?:pieces?|pcs|units?|kg)?\b",
    r"\bmanufacturer\b", r"\bsupplier\b", r"\bdistributor\b",
    r"\bregular\s+supply\b", r"\bmonthly\s+(?:order|supply)\b",
    r"\bcontract\b", r"\btender\b",
    # Hindi (Devanagari)
    r"थोक", r"बड़ा\s*ऑर्डर", r"माल\s*चाहिए", r"सामान\s*चाहिए",
    r"खेप", r"डीलर", r"व्यापारी", r"\d+\s*किलो", r"\d+\s*टन",
    # Marathi (Devanagari)
    r"घाऊक", r"मोठी\s*ऑर्डर", r"माल\s*हवे", r"नियमित\s*पुरवठा",
    r"\d+\s*किलो", r"मोठ्या\s*प्रमाणात",
    # Hinglish (Roman script)
    r"\bbulk\s*(?:mein|mai|me)\b", r"\bwholesale\s*(?:chahiye|chahie)\b",
    r"\bthok\b", r"\b(?:500|1000|2000|5000)\s*(?:kg|pcs|pieces|units)\b",
    r"\bbada\s*order\b", r"\bmaal\s*chahiye\b", r"\bsupply\s*chahiye\b",
    r"\bregular\s*order\b", r"\bmonthly\s*order\b",
}

_PRICE_QUERY_PATTERNS = {
    # English
    r"\bprice\b", r"\brate\b", r"\bcost\b", r"\bquote\b", r"\bquotation\b",
    r"\bhow\s+much\b", r"\bwhat\s+is\s+the\s+(?:price|rate|cost)\b",
    r"\bminimum\s+order\b", r"\bmoo\b", r"\bmoq\b",
    r"\bdiscount\b", r"\boffer\b", r"\bpricing\b", r"\bper\s+(?:kg|piece|unit)\b",
    r"\bcheap\b", r"\baffordable\b", r"\bbudget\b", r"\btariff\b",
    # Hindi (Devanagari)
    r"कीमत", r"दाम", r"भाव", r"क्या\s*रेट", r"कितने\s*में",
    r"कितना\s*पैसा", r"मूल्य", r"छूट", r"डिस्काउंट", r"सस्ता",
    r"रेट\s*बताओ", r"दाम\s*बताओ",
    # Marathi (Devanagari)
    r"किंमत", r"भाव", r"किती\s*रुपये", r"दर\s*काय", r"सांगा\s*किंमत",
    r"किती\s*आहे", r"स्वस्त", r"सवलत", r"किंमत\s*सांगा",
    # Hinglish (Roman script)
    r"\brate\s*(?:kya|batao|bata|do|do)\b", r"\bkitna\s*(?:rate|price|cost|paisa)\b",
    r"\bkitne\s*(?:mein|me|mai)\b", r"\bpaisa\s*kitna\b",
    r"\bdaam\s*(?:kya|batao|bata)\b", r"\bbhav\s*(?:kya|batao)\b",
    r"\bprice\s*(?:batao|kya|do)\b", r"\bsasta\b", r"\bdiscount\s*(?:milega|do)\b",
    r"\bkya\s*rate\b",
}

_COMPLAINT_PATTERNS = {
    # English
    r"\bcomplaint\b", r"\bproblem\b", r"\bissue\b", r"\bdefect(?:ive)?\b",
    r"\bnot\s+working\b", r"\bbroke\b", r"\bbroken\b", r"\bdamaged\b",
    r"\bnot\s+received\b", r"\bnot\s+delivered\b", r"\blate\s+delivery\b",
    r"\bwrong\s+(?:item|product|order)\b", r"\bpoor\s+quality\b",
    r"\bdisappointed\b", r"\bfrustrated\b", r"\brefund\b", r"\breplace\b",
    r"\bescalate\b", r"\bcheated\b", r"\bfraud\b", r"\bcheating\b",
    # Hindi (Devanagari)
    r"शिकायत", r"समस्या", r"खराब", r"टूटा", r"नहीं\s*आया", r"देरी",
    r"गलत\s*(?:सामान|माल|आइटम)", r"वापस", r"रिफंड", r"धोखा",
    r"बेकार", r"घटिया", r"काम\s*नहीं\s*कर", r"नाराज",
    # Marathi (Devanagari)
    r"तक्रार", r"समस्या", r"खराब", r"मिळाले\s*नाही", r"उशीर",
    r"चुकीचे\s*(?:सामान|उत्पादन)", r"परत", r"रिफंड", r"फसवणूक",
    r"बिघडलेले", r"काम\s*करत\s*नाही",
    # Hinglish (Roman script)
    r"\bshikayat\b", r"\bproblem\s*(?:hai|ho\s*gayi|hua)\b",
    r"\bkharab\b", r"\btoota\b", r"\bnahi\s*aaya\b", r"\bpahuncha\s*nahi\b",
    r"\bgalat\s*(?:maal|item|product)\b", r"\bwapas\b", r"\brefund\s*(?:chahiye|do)\b",
    r"\bdhokha\b", r"\bbakwaas\b", r"\bbekar\b", r"\bghanta\b",
    r"\bdelivery\s*(?:nahi|late|slow)\b",
}

_SPAM_PATTERNS = {
    # Universal spam signals
    r"\b(?:click\s+here|click\s+now)\b",
    r"\b(?:free|win|winner|won|congratulations|congrats)\b.*\b(?:prize|gift|reward|iphone|cash)\b",
    r"\b(?:limited\s+time\s+offer|act\s+now|hurry|expires?\s+soon)\b",
    r"\b(?:earn\s+money|make\s+money\s+fast|work\s+from\s+home)\b",
    r"\b(?:weight\s+loss|lose\s+\d+\s*kg|slim|fat\s+burner)\b",
    r"https?://bit\.ly|https?://tinyurl",
    r"\b(?:casino|gambling|lottery|bet|betting)\b",
    r"\b(?:sex|porn|nude|xxx)\b",
    r"\b(?:mlm|pyramid|network\s+marketing)\b",
    r"\b(?:guaranteed|100%\s+returns?|no\s+risk)\b.*\b(?:invest|profit)\b",
    # Repetitive/incoherent text signal
    r"(.)\1{5,}",
    r"([a-zA-Z]{1,3}\s?){10,}",   # too many short random tokens
}

# ── Compiled regex cache ───────────────────────────────────────────────────────
def _compile(patterns):
    return [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]

_COMPILED = {
    "bulk_order":  _compile(_BULK_ORDER_PATTERNS),
    "price_query": _compile(_PRICE_QUERY_PATTERNS),
    "complaint":   _compile(_COMPLAINT_PATTERNS),
    "spam":        _compile(_SPAM_PATTERNS),
}

_TRANSFORMER_PIPELINE = None  # lazy-loaded


def _rule_based_classify(text: str) -> Dict[str, float]:
    """
    Returns raw match scores for each intent (higher = more signals).
    """
    scores = {intent: 0.0 for intent in _COMPILED}
    for intent, patterns in _COMPILED.items():
        hits = sum(1 for p in patterns if p.search(text))
        scores[intent] = hits
    return scores


def _normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    """Convert raw hit counts to probabilities (softmax-like)."""
    total = sum(raw.values())
    if total == 0:
        return {k: 0.25 for k in raw}
    return {k: v / total for k, v in raw.items()}


def _transformer_classify(text: str) -> Dict[str, float]:
    """Zero-shot classification using XLM-RoBERTa (lazy-loaded)."""
    global _TRANSFORMER_PIPELINE
    if _TRANSFORMER_PIPELINE is None:
        from transformers import pipeline as hf_pipeline
        from config.settings import TRANSFORMER_MODEL, TRANSFORMER_CANDIDATE_LABELS
        _TRANSFORMER_PIPELINE = hf_pipeline(
            "zero-shot-classification",
            model=TRANSFORMER_MODEL,
        )
    from config.settings import TRANSFORMER_CANDIDATE_LABELS
    label_map = {
        "price query": "price_query",
        "bulk order":  "bulk_order",
        "complaint":   "complaint",
        "spam":        "spam",
    }
    result = _TRANSFORMER_PIPELINE(text, TRANSFORMER_CANDIDATE_LABELS)
    return {label_map[l]: s for l, s in zip(result["labels"], result["scores"])}


def classify_intent(text: str, lang: str = "en") -> Tuple[str, float, Dict[str, float]]:
    """
    Classify the intent of *text*.

    Returns:
        (intent, confidence, all_scores)
        intent     → one of price_query | bulk_order | complaint | spam
        confidence → float 0-1
        all_scores → dict of intent → probability
    """
    from config.settings import USE_TRANSFORMER_FALLBACK

    raw_scores  = _rule_based_classify(text)
    norm_scores = _normalize_scores(raw_scores)

    best_intent = max(norm_scores, key=norm_scores.get)
    best_conf   = norm_scores[best_intent]
    total_hits  = sum(raw_scores.values())

    # If confidence is low AND transformer fallback is enabled → use transformer
    if USE_TRANSFORMER_FALLBACK and (total_hits == 0 or best_conf < 0.50):
        try:
            norm_scores = _transformer_classify(text)
            best_intent = max(norm_scores, key=norm_scores.get)
            best_conf   = norm_scores[best_intent]
        except Exception:
            pass  # fall back silently

    # If no pattern matched at all, classify as spam with low confidence
    if total_hits == 0:
        return ("spam", 0.35, {k: 0.25 for k in norm_scores})

    return (best_intent, round(best_conf, 3), norm_scores)
