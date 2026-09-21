"""
LeadLens — Preprocessor
Handles language detection, script normalization, and text cleaning
for English, Hindi, Marathi, and Hinglish (code-mixed) inputs.
"""

import re
import unicodedata
from typing import Tuple

# ── Hinglish keyword signals (Roman-script Hindi/Marathi) ─────────────────────
_HINGLISH_SIGNALS = {
    # Common Hindi/Marathi words written in Roman script
    "chahiye", "kya", "kitna", "kitne", "batao", "bata", "mujhe", "humko",
    "hum", "mein", "hai", "hain", "nahi", "nai", "aur", "bhi", "toh",
    "agar", "lekin", "par", "ke", "ka", "ki", "ko", "se", "mera", "meri",
    "yaar", "bhai", "boss", "sir", "didi", "bhaiya", "ji", "aap", "tum",
    "hoga", "karo", "karna", "dena", "lena", "milega", "milna", "bhejo",
    "paisa", "paise", "rupaye", "rate", "dam", "daam", "sasta", "mehenga",
    "order", "maal", "saman", "cheez", "product", "item",
    # Marathi Roman
    "mhanje", "ahe", "nahi", "sangal", "sanga", "aahe", "mala", "tumhi",
    "aamhi", "tyanche", "kiti", "kasa", "kase", "tyala", "tyla",
    # Common mixing patterns
    "kab", "kahan", "kyun", "kaisa", "acha", "theek", "thik", "bilkul",
    "zaroor", "pakka", "confirm", "ho", "gaya", "gaye", "raha", "rahi",
}

# ── Devanagari Unicode range ──────────────────────────────────────────────────
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# ── Emoji / special symbol strip ─────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# ── URL / email strip ─────────────────────────────────────────────────────────
_URL_RE    = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE  = re.compile(r'\S+@\S+\.\S+')
_PHONE_RE  = re.compile(r'\b(?:\+91[\s-]?)?[6-9]\d{9}\b')          # Indian mobile

# ── Repeated char normalization (e.g. "hellooooo" → "hello") ─────────────────
_REPEAT_RE = re.compile(r'(.)\1{3,}')


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the primary language of *text*.
    Returns a (lang_code, confidence) tuple.
    Lang codes: 'en', 'hi', 'mr', 'hinglish', 'unknown'
    """
    text_stripped = text.strip()
    if not text_stripped:
        return ("unknown", 0.0)

    # ── 1. Check for Devanagari script ───────────────────────────────────────
    devanagari_chars = len(_DEVANAGARI_RE.findall(text_stripped))
    total_alpha = max(1, sum(c.isalpha() for c in text_stripped))
    deva_ratio = devanagari_chars / total_alpha

    if deva_ratio > 0.6:
        # Distinguish Hindi vs Marathi using script-level markers
        # Marathi-specific characters: ळ(U+0933), ञ(U+091E), काही markers
        marathi_markers = len(re.findall(r'[\u0933\u091E\u091F\u092F\u093E]', text_stripped))
        # Simple heuristic: if common Marathi endings present, lean Marathi
        marathi_words = re.findall(
            r'\b(?:आहे|नाही|आम्ही|तुम्ही|मला|किती|सांगा|द्या|घ्या|करा|येईल|असेल|ऑर्डर)\b',
            text_stripped
        )
        if len(marathi_words) >= 1 or marathi_markers > 3:
            return ("mr", min(0.95, 0.75 + 0.05 * len(marathi_words)))
        return ("hi", min(0.95, 0.70 + 0.05 * devanagari_chars))

    # ── 2. Check for Hinglish (Roman-script but Indian words) ────────────────
    tokens_lower = set(re.findall(r'\b[a-zA-Z]+\b', text_stripped.lower()))
    hinglish_hits = tokens_lower & _HINGLISH_SIGNALS
    if len(hinglish_hits) >= 1:
        # Could be pure Hinglish or English with some loanwords
        hit_ratio = len(hinglish_hits) / max(1, len(tokens_lower))
        if hit_ratio >= 0.10:
            return ("hinglish", min(0.90, 0.55 + hit_ratio))

    # ── 3. Try langdetect for remaining cases ────────────────────────────────
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        lang = detect(text_stripped)
        # Map langdetect codes to our codes
        mapping = {"hi": "hi", "mr": "mr", "en": "en"}
        return (mapping.get(lang, "en"), 0.80)
    except Exception:
        pass

    # ── 4. Default to English ─────────────────────────────────────────────────
    return ("en", 0.60)


def clean_text(text: str) -> str:
    """
    Clean and normalize raw inquiry text.
    Preserves Devanagari characters; strips noise.
    """
    if not text:
        return ""

    # Strip URLs, emails
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)

    # Normalize unicode (NFC form keeps Devanagari intact)
    text = unicodedata.normalize("NFC", text)

    # Strip emojis but keep text content
    text = _EMOJI_RE.sub(" ", text)

    # Normalize repeated chars
    text = _REPEAT_RE.sub(r'\1\1', text)

    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_numbers(text: str) -> str:
    """
    Normalize various number representations to Arabic numerals.
    Handles: Hindi/Marathi Devanagari digits, k/K shorthand.
    """
    # Devanagari digit map — must happen FIRST before regex replacements
    deva_digits = str.maketrans('०१२३४५६७८९', '0123456789')
    text = text.translate(deva_digits)

    # Expand k/K shorthand: 5k → 5000, 2.5k → 2500
    # (only if k is not immediately followed by g/m/s — to avoid matching kg)
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*[kK](?![gGmMsSlL])\b',
        lambda m: str(int(float(m.group(1)) * 1000)),
        text
    )
    # Expand lakh: 1 lakh / 1 lac (NOT bare 'L' to avoid matching unit abbreviations)
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:lakh|lac)\b',
        lambda m: str(int(float(m.group(1)) * 100_000)),
        text,
        flags=re.IGNORECASE
    )
    return text


def preprocess(text: str) -> dict:
    """
    Full preprocessing pipeline.
    Returns a dict with:
      - original     : original text
      - cleaned      : cleaned text
      - language     : detected language code
      - lang_conf    : language detection confidence (0-1)
      - normalized   : number-normalized cleaned text
    """
    original  = text
    cleaned   = clean_text(text)
    lang, conf = detect_language(cleaned)
    normalized = normalize_numbers(cleaned)

    return {
        "original":   original,
        "cleaned":    cleaned,
        "language":   lang,
        "lang_conf":  round(conf, 3),
        "normalized": normalized,
    }
