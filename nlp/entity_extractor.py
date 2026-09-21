"""
LeadLens — Entity Extractor
Extracts key entities from customer inquiries:
  - product   : what they want to buy/query
  - quantity  : amount (with unit)
  - location  : Indian city/state mentioned
  - budget    : monetary value / price range
"""

import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Optional

# ── Quantity patterns ──────────────────────────────────────────────────────────
_QTY_RE = re.compile(
    r"""
    (?P<number>\d+(?:[.,]\d+)?)        # digits with optional decimal
    \s*                                 # optional space
    (?P<unit>
        kg(?:s)?|kilogram(?:s)?|kilo(?:s)?|  # weight - metric
        ton(?:s)?|tonne(?:s)?|               # weight - heavy
        quintal(?:s)?|                       # weight - Indian
        gram(?:s)?|gm(?:s)?|                 # weight - small
        liter(?:s)?|litre(?:s)?|lt(?:s)?|    # volume
        ml|milliliter(?:s)?|                  # volume - small
        piece(?:s)?|pcs|pc|                   # count
        unit(?:s)?|                           # count
        no(?:s)?|number(?:s)?|               # count
        dozen(?:s)?|dz|                       # count - dozen
        gross|                               # count - 144
        box(?:es)?|carton(?:s)?|packet(?:s)?|bag(?:s)?|  # packaging
        meter(?:s)?|metre(?:s)?|mtr(?:s)?|mt|  # length
        foot|feet|ft|yard(?:s)?|             # length - imperial
        roll(?:s)?|sheet(?:s)?|              # fabric/paper
        set(?:s)?|pair(?:s)?|                # grouped items
        किलो|किग्रा|ग्राम|लीटर|नग|दर्जन|मीटर  # Hindi/Marathi units
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Budget / Price patterns ────────────────────────────────────────────────────
_BUDGET_RE = re.compile(
    r"""
    (?:
        ₹\s*(?P<amt1>\d+(?:[,\d]*\d)?(?:\.\d+)?)         |  # ₹5000
        Rs\.?\s*(?P<amt2>\d+(?:[,\d]*\d)?(?:\.\d+)?)     |  # Rs 5000
        INR\s*(?P<amt3>\d+(?:[,\d]*\d)?(?:\.\d+)?)       |  # INR 5000
        (?P<amt4>\d+(?:[,\d]*\d)?)\s*(?:rupees?|rupaiye|रुपये|रुपए|रु)  # 5000 rupees
    )
    (?:                                                       # optional range
        \s*[-–to]+\s*
        (?:₹|Rs\.?|INR)?\s*
        (?P<amt_hi>\d+(?:[,\d]*\d)?(?:\.\d+)?)
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Indian city / state lookup ─────────────────────────────────────────────────
def _load_locations():
    from config.settings import INDIAN_CITIES
    # Build a regex that matches whole words
    pattern = r'\b(?:' + '|'.join(re.escape(c) for c in sorted(INDIAN_CITIES, key=len, reverse=True)) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

_LOCATION_RE: Optional[re.Pattern] = None

def _get_location_re() -> re.Pattern:
    global _LOCATION_RE
    if _LOCATION_RE is None:
        _LOCATION_RE = _load_locations()
    return _LOCATION_RE

# Pin code pattern
_PINCODE_RE = re.compile(r'\b[1-9][0-9]{5}\b')


def extract_quantity(text: str) -> Optional[Dict]:
    """Extract the first prominent quantity mention."""
    match = _QTY_RE.search(text)
    if match:
        number = match.group("number").replace(",", "")
        unit   = match.group("unit")
        return {
            "value": float(number),
            "unit":  unit,
            "raw":   match.group(0).strip(),
        }
    # Bare large number without unit (likely a quantity in context)
    bare = re.search(r'\b(\d{3,})\b', text)
    if bare:
        num = int(bare.group(1))
        if num >= 50:  # ≥50 likely a quantity
            return {"value": float(num), "unit": "units", "raw": bare.group(0)}
    return None


def extract_budget(text: str) -> Optional[Dict]:
    """Extract budget / price range from text."""
    match = _BUDGET_RE.search(text)
    if not match:
        return None
    # pick first non-None amount group
    low = next(
        (match.group(g) for g in ("amt1", "amt2", "amt3", "amt4") if match.group(g)),
        None
    )
    high = match.group("amt_hi")
    if low is None:
        return None
    low_val = float(low.replace(",", ""))
    result = {"low": low_val, "raw": match.group(0).strip()}
    if high:
        result["high"] = float(high.replace(",", ""))
        result["display"] = f"₹{int(low_val):,} – ₹{int(result['high']):,}"
    else:
        result["display"] = f"₹{int(low_val):,}"
    return result


def extract_location(text: str) -> Optional[str]:
    """Extract first Indian city or state mention."""
    loc_re = _get_location_re()
    match = loc_re.search(text)
    if match:
        return match.group(0).title()
    # Fallback: check for pin code
    pin = _PINCODE_RE.search(text)
    if pin:
        return f"PIN {pin.group(0)}"
    return None


def extract_product(text: str) -> Optional[str]:
    """
    Extract product name from text using product keyword catalog.
    Returns the matched product keyword (most specific match first).
    """
    from config.settings import PRODUCT_KEYWORDS
    text_lower = text.lower()

    # Score keywords by specificity (longer = more specific)
    matches = []
    for kw in PRODUCT_KEYWORDS:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE):
            matches.append(kw)

    if not matches:
        return None
    # Return longest match (most specific)
    return max(matches, key=len).title()


def extract_entities(text: str) -> Dict:
    """
    Run full entity extraction on *text*.

    Returns dict:
        product  : str | None
        quantity : dict | None  → {value, unit, raw}
        location : str | None
        budget   : dict | None  → {low, high?, display, raw}
    """
    return {
        "product":  extract_product(text),
        "quantity": extract_quantity(text),
        "location": extract_location(text),
        "budget":   extract_budget(text),
    }
