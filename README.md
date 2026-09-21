# 🔍 LeadLens — Multilingual NLP Lead Intelligence

> **Analyze customer inquiries in English, Hindi, Marathi & Hinglish.  
> Classify intent, extract key entities, and prioritize leads — all in real-time.**

---

## Features

| Feature | Details |
|---------|---------|
| 🌐 **4 Languages** | English · Hindi (Devanagari) · Marathi · Hinglish (code-mixed) |
| 🧠 **Intent Classification** | Price Query · Bulk Order · Complaint · Spam |
| 🏷️ **Entity Extraction** | Product · Quantity · Location · Budget |
| 📊 **Lead Scoring** | 0–100 composite score → Hot / Warm / Cold tiers |
| ⚡ **Real-time Dashboard** | Streamlit UI with charts, filters, and bulk upload |

---

## Project Structure

```
NLP/
├── app/
│   └── dashboard.py        # Streamlit dashboard (main UI)
├── nlp/
│   ├── preprocessor.py     # Language detection & text cleaning
│   ├── intent_classifier.py# Rule-based + optional XLM-RoBERTa classifier
│   ├── entity_extractor.py # Regex-based entity extraction
│   ├── lead_scorer.py      # Composite scoring & recommended actions
│   └── pipeline.py         # Full pipeline orchestrator
├── data/
│   ├── sample_inquiries.csv# 48 multilingual sample inquiries
│   └── leads_store.json    # Persisted lead results
├── config/
│   └── settings.py         # Thresholds, labels, model config
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard
```bash
cd NLP
streamlit run app/dashboard.py
```

The dashboard opens at **http://localhost:8501**

### 3. (Optional) Run the pipeline CLI test
```bash
python nlp/pipeline.py
```

---

## Configuration

Edit **`config/settings.py`** to tune:

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_TRANSFORMER_FALLBACK` | `False` | Enable XLM-RoBERTa for ambiguous cases |
| `PRIORITY_HOT_THRESHOLD` | `70` | Min score for Hot priority |
| `PRIORITY_WARM_THRESHOLD` | `40` | Min score for Warm priority |
| `INTENT_WEIGHTS` | See file | Per-intent scoring weights |

---

## Scoring Formula

```
score = intent_score (40) + entity_richness (30) + language_clarity (15) + specificity (15)
```

| Component | Max | Logic |
|-----------|-----|-------|
| **Intent score** | 40 | bulk_order=1.0, price_query=0.7, complaint=0.3, spam=0.0 × confidence |
| **Entity richness** | 30 | 7.5 pts per extracted entity (product, qty, location, budget) |
| **Language clarity** | 15 | Detection confidence × 15 |
| **Specificity** | 15 | Both qty+budget: 15 pts · One: 9 pts · Product only: 4.5 pts |

**Priority Tiers:**
- 🔴 **Hot** → score ≥ 70 (respond within 1 hour)
- 🟡 **Warm** → score 40–69 (respond within 24 hours)  
- 🔵 **Cold** → score < 40 (low priority)

---

## Sample Inquiries

| Text | Language | Intent | Priority |
|------|----------|--------|----------|
| `Mujhe 500 kg cotton chahiye, rate batao, Mumbai` | Hinglish | Bulk Order | 🔴 Hot |
| `आपल्या उत्पादनाची किंमत किती आहे?` | Marathi | Price Query | 🟡 Warm |
| `Yaar delivery bahut slow hai, 2 hafte ho gaye` | Hinglish | Complaint | 🔵 Cold |
| `Click here to WIN iPhone!` | English | Spam | 🔵 Cold |
| `हमें हर महीने 2 टन चावल चाहिए, ₹80000 budget` | Hindi | Bulk Order | 🔴 Hot |

---

## Tech Stack

- **NLP**: Rule-based multilingual patterns + optional XLM-RoBERTa (HuggingFace)
- **Dashboard**: Streamlit + Plotly
- **Data**: pandas + JSON
- **Languages**: Python 3.10+
