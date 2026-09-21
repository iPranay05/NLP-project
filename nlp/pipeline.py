"""
LeadLens — NLP Pipeline
Orchestrates the full analysis of a customer inquiry:
  preprocess → intent → entities → score → action
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import uuid
from datetime import datetime
from typing import Dict

from nlp.preprocessor      import preprocess
from nlp.intent_classifier  import classify_intent
from nlp.entity_extractor   import extract_entities
from nlp.lead_scorer        import compute_score, get_recommended_action


def analyze(text: str) -> Dict:
    """
    Run the full LeadLens pipeline on a single customer inquiry.

    Returns a result dict with:
        id, timestamp, original_text, language, lang_conf,
        intent, intent_confidence, intent_scores,
        entities, score, priority, score_breakdown, recommended_action
    """
    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    pre = preprocess(text)

    # ── Step 2: Intent Classification ─────────────────────────────────────────
    intent, conf, all_scores = classify_intent(
        pre["normalized"],
        lang=pre["language"],
    )

    # ── Step 3: Entity Extraction ──────────────────────────────────────────────
    entities = extract_entities(pre["normalized"])

    # ── Step 4: Lead Scoring ───────────────────────────────────────────────────
    score, priority, breakdown = compute_score(
        intent=intent,
        confidence=conf,
        entities=entities,
        lang=pre["language"],
        lang_conf=pre["lang_conf"],
    )

    # ── Step 5: Recommended Action ─────────────────────────────────────────────
    action = get_recommended_action(intent, priority, entities, lang=pre["language"])

    return {
        "id":                 str(uuid.uuid4())[:8],
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "original_text":      pre["original"],
        "language":           pre["language"],
        "lang_conf":          pre["lang_conf"],
        "intent":             intent,
        "intent_confidence":  conf,
        "intent_scores":      {k: round(v, 3) for k, v in all_scores.items()},
        "entities":           entities,
        "score":              score,
        "priority":           priority,
        "score_breakdown":    breakdown,
        "recommended_action": action,
    }


def analyze_batch(texts: list) -> list:
    """Analyze a list of customer inquiry strings."""
    return [analyze(t) for t in texts if t and t.strip()]


# ── CLI test runner ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_cases = [
        "Mujhe 500 kg cotton chahiye, best rate batao, Mumbai delivery",
        "आपल्या उत्पादनाची किंमत किती आहे? पुणे येथे डिलिव्हरी होईल का?",
        "Yaar delivery bahut slow hai, 2 hafte ho gaye aur maal abhi nahi aaya",
        "Click here to win iPhone! Limited time offer!",
        "What is the wholesale price for 1000 pieces of silk saree?",
        "हमें हर महीने 2 टन चावल चाहिए, दिल्ली डिलीवरी, बजट ₹80000",
        "Product quality kharab hai, refund chahiye",
        "तुमचा माल खूप महाग आहे, सवलत द्याल का?",
        "I need 200 kg of turmeric powder, budget is Rs 15000, Pune",
        "Bhai mere ko 50 dozen cotton kurta chahiye wholesale mein",
    ]

    print("\n" + "=" * 70)
    print("  LeadLens — Pipeline Integration Test")
    print("=" * 70)

    for i, text in enumerate(test_cases, 1):
        result = analyze(text)
        ent = result["entities"]
        print(f"\n[{i}] {result['original_text'][:60]}{'...' if len(text) > 60 else ''}")
        print(f"     Language : {result['language']} (conf={result['lang_conf']})")
        print(f"     Intent   : {result['intent']} (conf={result['intent_confidence']})")
        print(f"     Product  : {ent.get('product')}")
        print(f"     Quantity : {ent['quantity']['raw'] if ent.get('quantity') else None}")
        print(f"     Budget   : {ent['budget']['display'] if ent.get('budget') else None}")
        print(f"     Location : {ent.get('location')}")
        print(f"     Score    : {result['score']} -> {result['priority']}")
        print(f"     Action   : {result['recommended_action'][:80]}")
