# ─────────────────────────────────────────────
#  LeadLens — Central Configuration
# ─────────────────────────────────────────────

# ── Model Settings ───────────────────────────
# Set to True to enable XLM-RoBERTa zero-shot for ambiguous cases.
# First run will download ~1.1 GB from HuggingFace. Requires internet.
# Set to False for 100% offline, instant rule-based pipeline.
USE_TRANSFORMER_FALLBACK = False

TRANSFORMER_MODEL = "joeddav/xlm-roberta-large-xnli"   # zero-shot classification
TRANSFORMER_CANDIDATE_LABELS = ["price query", "bulk order", "complaint", "spam"]

# ── Intent Classes ───────────────────────────
INTENTS = {
    "price_query": {
        "label": "Price Query",
        "emoji": "💰",
        "color": "#6C63FF",
        "response_sla": "24 hours",
    },
    "bulk_order": {
        "label": "Bulk Order",
        "emoji": "📦",
        "color": "#00D4AA",
        "response_sla": "1 hour",
    },
    "complaint": {
        "label": "Complaint",
        "emoji": "⚠️",
        "color": "#FF6B6B",
        "response_sla": "2 hours",
    },
    "spam": {
        "label": "Spam",
        "emoji": "🚫",
        "color": "#888888",
        "response_sla": "N/A",
    },
}

# ── Lead Priority Tiers ───────────────────────
PRIORITY_HOT_THRESHOLD   = 70   # score >= 70 → Hot
PRIORITY_WARM_THRESHOLD  = 40   # score >= 40 → Warm
# score < 40 → Cold

PRIORITY_CONFIG = {
    "Hot":  {"emoji": "🔴", "color": "#FF4757", "badge": "bg-hot",  "sla": "Respond within 1 hour"},
    "Warm": {"emoji": "🟡", "color": "#FFA502", "badge": "bg-warm", "sla": "Respond within 24 hours"},
    "Cold": {"emoji": "🔵", "color": "#5352ED", "badge": "bg-cold", "sla": "Low priority"},
}

# ── Scoring Weights ───────────────────────────
SCORE_WEIGHTS = {
    "intent":      40,   # max 40 pts
    "entities":    30,   # max 30 pts (7.5 per entity, up to 4)
    "language":    15,   # max 15 pts (language detection confidence)
    "specificity": 15,   # max 15 pts (both qty + budget present)
}

INTENT_WEIGHTS = {
    "bulk_order":  1.00,
    "price_query": 0.70,
    "complaint":   0.30,
    "spam":        0.00,
}

# ── Supported Languages ───────────────────────
LANGUAGES = {
    "en":      {"name": "English",  "emoji": "🇬🇧"},
    "hi":      {"name": "Hindi",    "emoji": "🇮🇳"},
    "mr":      {"name": "Marathi",  "emoji": "🟠"},
    "hinglish":{"name": "Hinglish", "emoji": "🔀"},
    "unknown": {"name": "Unknown",  "emoji": "❓"},
}

# ── Indian Cities for Location Extraction ─────
INDIAN_CITIES = [
    "mumbai", "delhi", "pune", "bangalore", "bengaluru", "hyderabad",
    "ahmedabad", "chennai", "kolkata", "surat", "jaipur", "lucknow",
    "kanpur", "nagpur", "indore", "thane", "bhopal", "visakhapatnam",
    "vadodara", "firozabad", "ludhiana", "agra", "nashik", "meerut",
    "rajkot", "varanasi", "amritsar", "allahabad", "prayagraj", "patna",
    "coimbatore", "madurai", "gurgaon", "gurugram", "noida", "chandigarh",
    "aurangabad", "solapur", "jabalpur", "gwalior", "vijayawada", "jodhpur",
    "raipur", "kochi", "ernakulam", "srinagar", "ranchi", "guwahati",
    "bhubaneswar", "pimpri", "navi mumbai", "kolhapur", "sangli", "satara",
    # State names
    "maharashtra", "gujarat", "rajasthan", "punjab", "uttar pradesh",
    "madhya pradesh", "karnataka", "tamilnadu", "tamil nadu", "kerala",
    "west bengal", "odisha",
]

# ── Product Catalog (seed keywords) ───────────
PRODUCT_KEYWORDS = {
    # Textiles
    "cotton", "silk", "linen", "polyester", "nylon", "jute", "wool",
    "fabric", "kapda", "saree", "sari", "suit", "dupatta", "lehenga",
    "kurta", "shirt", "kurti", "blouse", "denim", "chiffon", "georgette",
    # Electronics
    "mobile", "phone", "laptop", "tablet", "speaker", "earphone", "headphone",
    "charger", "cable", "adapter", "led", "bulb", "fan", "cooler", "ac",
    "refrigerator", "fridge", "washing machine", "camera", "cctv",
    # Food & Spices
    "rice", "wheat", "sugar", "salt", "spice", "masala", "turmeric", "haldi",
    "chilli", "mirchi", "coriander", "dhania", "cumin", "jeera", "oil", "ghee",
    "dal", "lentil", "flour", "atta", "maida", "sooji", "suji", "pulses",
    "tur", "moong", "chana", "rajma", "tea", "chai", "coffee", "namkeen",
    # Hardware & Tools
    "pipe", "wire", "cable", "screw", "bolt", "nut", "paint", "cement",
    "tile", "brick", "rod", "iron", "steel", "plastic", "pvc", "fitting",
    # Stationery & Packaging
    "paper", "box", "carton", "bag", "packet", "bottle", "container",
    "label", "sticker", "pen", "book", "notebook",
}

# ── Data Paths ────────────────────────────────
DATA_DIR           = "data"
LEADS_STORE_FILE   = "data/leads_store.json"
SAMPLE_CSV_FILE    = "data/sample_inquiries.csv"
