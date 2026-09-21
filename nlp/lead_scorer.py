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


def get_recommended_action(intent: str, priority: str, entities: Dict) -> str:
    """Generate a short recommended action string for the sales team."""
    if intent == "spam":
        return "Mark as spam. No action required."

    product = entities.get("product")
    qty     = entities.get("quantity")
    budget  = entities.get("budget")
    loc     = entities.get("location")

    parts = []
    if intent == "bulk_order":
        parts.append("Send bulk pricing catalogue")
        if qty:
            parts.append(f"confirm availability for {qty['raw']}")
        if loc:
            parts.append(f"check delivery to {loc}")
    elif intent == "price_query":
        parts.append("Share product price list")
        if product:
            parts.append(f"focus on {product}")
        if budget:
            parts.append(f"budget noted: {budget['display']}")
    elif intent == "complaint":
        parts.append("Escalate to support team")
        parts.append("request order ID for resolution")

    if priority == "Hot":
        parts.insert(0, "🔴 URGENT — call within 1 hour")
    elif priority == "Warm":
        parts.insert(0, "🟡 Follow up within 24 hours")

    return ". ".join(parts) + "." if parts else "Review manually."
