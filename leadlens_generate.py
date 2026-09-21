"""
LeadLens synthetic dataset generator.

Outputs (same folder as this script):
  leadlens_dataset.csv   - one row per message, all labels as columns
  leadlens_ner.jsonl     - same messages with character-level entity spans + BIO tags

Languages : en, hinglish, hi (Devanagari), mr (Devanagari), mr_roman (Marathi in Latin script)
Intents   : price_query, bulk_order, complaint, spam
Entities  : PRODUCT, QTY, CITY, BUDGET   (URGENCY is a 0/1 flag, not a span)

The train/val/test split is TEMPLATE-DISJOINT: a test message never shares a
template with a training message. Scores on it are therefore harder than a
random split, which is what you want. It is still synthetic, so always
validate on real messages too.

Usage:  python leadlens_generate.py [rows_per_language_intent_cell]   (default 90)
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

SEED = 42
PER_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 90
OUT = Path(__file__).parent
random.seed(SEED)

LANGS = ["en", "hinglish", "hi", "mr", "mr_roman"]
INTENTS = ["price_query", "bulk_order", "complaint", "spam"]
ROMAN = {"en", "hinglish", "mr_roman"}

# ---------------------------------------------------------------- vocab ----
CITIES = {
    "roman": ["Pune", "Mumbai", "Thane", "Kalyan", "Nashik", "Nagpur", "Aurangabad", "Kolhapur", "Solapur",
              "Satara", "Surat", "Indore", "Bhopal", "Hyderabad", "Bengaluru", "Ahmedabad", "Goa", "Belgaum",
              "Sangli", "Jalgaon", "Latur", "Nanded", "Ratnagiri", "Vasai", "Panvel", "Navi Mumbai"],
    "hi": ["पुणे", "मुंबई", "ठाणे", "कल्याण", "नाशिक", "नागपुर", "औरंगाबाद", "कोल्हापुर", "सोलापुर", "सातारा",
           "सूरत", "इंदौर", "भोपाल", "हैदराबाद", "बेंगलुरु", "अहमदाबाद", "गोवा", "बेलगाम", "सांगली", "जलगांव",
           "लातूर", "नांदेड़", "रत्नागिरी", "वसई", "पनवेल", "नवी मुंबई"],
    "mr": ["पुणे", "मुंबई", "ठाणे", "कल्याण", "नाशिक", "नागपूर", "औरंगाबाद", "कोल्हापूर", "सोलापूर", "सातारा",
           "सुरत", "इंदूर", "भोपाळ", "हैदराबाद", "बेंगळुरू", "अहमदाबाद", "गोवा", "बेळगाव", "सांगली", "जळगाव",
           "लातूर", "नांदेड", "रत्नागिरी", "वसई", "पनवेल", "नवी मुंबई"],
}
CITY_KEY = {"en": "roman", "hinglish": "roman", "mr_roman": "roman", "hi": "hi", "mr": "mr"}

PRODUCTS = {
    "en": ["bus seats", "pushback seats", "reclining seats", "seat covers", "sleeper berths", "roof-top AC",
           "bus AC", "bus wrap", "vinyl branding", "exterior wrap", "LED lights", "interior LED lighting",
           "PVC flooring", "anti-skid flooring", "window curtains", "luggage racks", "exterior paint",
           "full body paint", "music system", "CCTV cameras", "ceiling panels"],
    "hinglish": ["bus seats", "pushback seats", "seat cover", "sleeper berth", "roof top AC", "bus wrap",
                 "vinyl branding", "LED lights", "flooring", "curtains", "luggage rack", "body paint",
                 "music system", "CCTV camera", "ceiling panel"],
    "hi": ["बस सीट", "पुशबैक सीट", "सीट कवर", "स्लीपर बर्थ", "रूफ टॉप AC", "बस रैप", "LED लाइट", "फ्लोरिंग",
           "पर्दे", "लगेज रैक", "बॉडी पेंट", "म्यूजिक सिस्टम", "CCTV कैमरा", "सीलिंग पैनल"],
    "mr": ["बस सीट", "पुशबॅक सीट", "सीट कव्हर", "स्लीपर बर्थ", "रूफ टॉप AC", "बस रॅप", "LED लाईट", "फ्लोअरिंग",
           "पडदे", "लगेज रॅक", "बॉडी पेंट", "म्युझिक सिस्टीम", "CCTV कॅमेरा", "सीलिंग पॅनल"],
    "mr_roman": ["bus seats", "pushback seats", "seat cover", "sleeper berth", "roof top AC", "bus wrap",
                 "LED lights", "flooring", "padde", "luggage rack", "body paint", "music system",
                 "CCTV camera", "ceiling panel"],
}

BUS_UNIT = {"en": lambda n: "bus" if n == 1 else "buses",
            "hinglish": lambda n: random.choice(["bus", "buses"]) if n > 1 else "bus",
            "hi": lambda n: random.choice(["बस", "बसों"]) if n > 1 else "बस",
            "mr": lambda n: random.choice(["बस", "बसेस"]) if n > 1 else "बस",
            "mr_roman": lambda n: random.choice(["bus", "buses"]) if n > 1 else "bus"}
ITEM_UNIT = {"en": ["seats", "sets", "units"], "hinglish": ["seats", "sets", "pieces"],
             "hi": ["सीट", "सेट", "पीस"], "mr": ["सीट", "सेट", "नग"], "mr_roman": ["seats", "sets", "nag"]}


def make_qty(lang, intent):
    if intent == "bulk_order":
        if random.random() < 0.7:
            n = random.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50])
            return f"{n} {BUS_UNIT[lang](n)}"
        n = random.choice([40, 50, 60, 80, 100, 120, 150, 200, 250, 300])
        return f"{n} {random.choice(ITEM_UNIT[lang])}"
    if random.random() < 0.6:
        n = random.choice([1, 1, 2, 3])
        return f"{n} {BUS_UNIT[lang](n)}"
    n = random.choice([1, 2, 4, 10, 20, 45])
    return f"{n} {random.choice(ITEM_UNIT[lang])}"


def trim(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def make_budget(lang, intent):
    amt = (random.choice([20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000])
           if intent == "price_query"
           else random.choice([500000, 800000, 1000000, 1500000, 2500000, 5000000, 10000000]))
    if amt >= 10000000:
        n = trim(amt / 10000000)
        return {"en": f"{n} crore", "hinglish": random.choice([f"{n} crore", f"{n} cr"]),
                "hi": f"{n} करोड़", "mr": f"{n} कोटी", "mr_roman": f"{n} koti"}[lang]
    if amt >= 100000:
        n = trim(amt / 100000)
        return {"en": random.choice([f"₹{n} lakh", f"{n} lakh", f"Rs {n} lakh"]),
                "hinglish": random.choice([f"{n} lakh", f"{n}L", f"Rs {n} lakh"]),
                "hi": random.choice([f"{n} लाख", f"₹{n} लाख"]),
                "mr": random.choice([f"{n} लाख", f"₹{n} लाख"]),
                "mr_roman": random.choice([f"{n} lakh", f"Rs {n} lakh", f"{n} lakh rupaye"])}[lang]
    k = amt // 1000
    return {"en": random.choice([f"{k}k", f"₹{amt:,}", f"Rs {amt}"]),
            "hinglish": random.choice([f"{k}k", f"{amt}", f"Rs {amt}"]),
            "hi": random.choice([f"{k} हज़ार", f"{amt} रुपये", f"₹{amt:,}"]),
            "mr": random.choice([f"{k} हजार", f"{amt} रुपये"]),
            "mr_roman": random.choice([f"{k} hajar", f"Rs {amt}"])}[lang]


# ------------------------------------------------------------ templates ----
# {P}=PRODUCT {Q}=QTY {C}=CITY {B}=BUDGET  (labelled)
# {L}{M}{E}{F}{D} = random numbers used in spam only (NOT labelled on purpose)
T = {
 "en": {
  "price_query": [
   "What is the price of {P} for a bus?",
   "Can you share the rate for {P}? Need it for {Q}.",
   "How much would {P} cost for {Q} in {C}?",
   "Please send a quotation for {P}. Budget is around {B}.",
   "Do you install {P} in {C}? What's the cost?",
   "Price list for {P} please",
   "Rate for {P}, my budget is {B}",
   "What would it cost to get {P} done on my bus? I'm in {C}.",
   "Quote for {Q} with {P}, budget {B}",
  ],
  "bulk_order": [
   "We want to order {P} for {Q}. Please share bulk pricing.",
   "I run a travel company in {C} and need {P} for {Q}. What discount can you offer?",
   "Looking to refurbish {Q} with {P}. Budget {B}. When can you start?",
   "Bulk order: {Q} need {P}. Delivery to {C} in 3 weeks.",
   "Our fleet of {Q} needs {P}. Can we schedule a visit to {C}?",
   "Need {P} for {Q}, total budget {B}. Please call me.",
   "Ordering {P} for {Q} in {C}, please confirm the wholesale rate.",
   "We are a school in {C}. Need {P} for {Q}. What is the best price?",
  ],
  "complaint": [
   "The {P} you installed is already damaged. This is unacceptable.",
   "I paid for {P} but the work is still incomplete after 2 weeks.",
   "Very poor finish on the {P}. Please fix it immediately.",
   "{P} stopped working within a week. I want a refund.",
   "Your team delayed my bus delivery in {C} by 10 days and nobody is answering my calls.",
   "The {P} does not match what was promised. Need someone to inspect it.",
   "Warranty claim for {P}. It broke after one month.",
  ],
  "spam": [
   "Get a personal loan up to {L} lakh in {M} minutes. Click the link now!",
   "Congratulations! You have won a free iPhone. Claim your prize.",
   "Boost your Instagram followers by {F}. Cheap rates, DM now.",
   "We offer SEO and website ranking services. Guaranteed first page on Google.",
   "Earn {E} per day working from home. WhatsApp us.",
   "Invest in crypto and double your money in {D} days.",
   "Dear sir, we are a supplier of cheap LED lights from China. Reply for a catalogue.",
   "Free credit card offer, no annual fee. Apply now.",
   "Buy 1 get 1 free on all sarees. Visit our store today!",
   "Get a loan for your bus business up to {L} lakh with zero documentation. Click now!",
   "Grow your travel business with guaranteed leads. Limited offer, contact us today.",
  ],
 },
 "hinglish": {
  "price_query": [
   "bhai {P} ka rate kya hai?",
   "{C} me {P} lagwane ka kitna kharcha aayega?",
   "please {P} ka quotation bhej do. budget {B} hai",
   "{Q} ke liye {P} ki price batao",
   "mujhe {P} ki price janni hai, budget around {B}",
   "kya aap {C} me {P} ka kaam karte ho? rate batao",
   "{P} ki price list bhejo",
   "sir {P} kitne ka padega {Q} me",
  ],
  "bulk_order": [
   "hume {Q} ke liye {P} chahiye. bulk me kya rate milega?",
   "meri travels company {C} me hai, {Q} me {P} lagwana hai. kitna discount milega?",
   "bulk order dena hai {Q} ka, {P} chahiye. budget {B}",
   "hamare paas {Q} hai, sabme {P} lagwana hai. please call karo",
   "{C} delivery ke liye {Q} me {P} chahiye, 3 weeks me",
   "{Q} ke liye {P} ka wholesale rate batao, total budget {B}",
  ],
  "complaint": [
   "jo {P} aapne lagaya tha wo kharab ho gaya. ye theek nahi hai",
   "maine {P} ke paise de diye lekin kaam abhi tak adhura hai",
   "{P} ki finishing bahut kharab hai, turant theek karo",
   "{P} ek hafte me hi band ho gaya. mujhe refund chahiye",
   "{C} me meri bus ki delivery 10 din late hui aur koi phone nahi utha raha",
   "warranty me {P} change karwana hai, ek mahine me toot gaya",
  ],
  "spam": [
   "{L} lakh tak personal loan pao {M} minute me. abhi click karo!",
   "congratulations! aapne free iPhone jeeta hai. abhi claim karo",
   "instagram followers {F} badhao, sasta rate. DM karo",
   "ghar baithe roz {E} kamao. WhatsApp karo",
   "crypto me invest karo aur {D} din me paisa double karo",
   "google pe first page guarantee wali SEO service lo",
   "sabhi sarees pe 1 ke saath 1 free. aaj hi dukaan aao",
   "bus business ke liye {L} lakh tak loan pao, koi documents nahi. abhi click karo",
   "apne travel business ke liye guaranteed leads pao. limited offer, aaj hi contact karo",
   "free credit card offer, koi annual fee nahi. abhi apply karo",
  ],
 },
 "hi": {
  "price_query": [
   "{P} का रेट क्या है?",
   "{C} में {P} लगवाने का खर्च कितना आएगा?",
   "कृपया {P} का कोटेशन भेजें। बजट {B} है।",
   "{Q} के लिए {P} की कीमत बताइए।",
   "मुझे {P} की कीमत जाननी है, बजट लगभग {B} है।",
   "क्या आप {C} में {P} का काम करते हैं? रेट बताइए।",
   "{P} का प्राइस लिस्ट भेज दीजिए",
  ],
  "bulk_order": [
   "हमें {Q} के लिए {P} चाहिए। थोक में क्या रेट मिलेगा?",
   "मेरी ट्रैवल कंपनी {C} में है, {Q} में {P} लगवाना है। कितना डिस्काउंट मिलेगा?",
   "{Q} का बल्क ऑर्डर देना है, {P} चाहिए। बजट {B}।",
   "हमारे पास {Q} हैं, सभी में {P} लगवाना है। कृपया कॉल करें।",
   "{C} डिलीवरी के लिए {Q} में {P} चाहिए, 3 हफ्ते में।",
   "{Q} के लिए {P} का थोक रेट बताइए, कुल बजट {B} है।",
  ],
  "complaint": [
   "आपने जो {P} लगाया था वो खराब हो गया है। यह ठीक नहीं है।",
   "मैंने {P} के पैसे दे दिए, लेकिन काम अभी तक अधूरा है।",
   "{P} की फिनिशिंग बहुत खराब है, तुरंत ठीक करवाइए।",
   "{P} एक हफ्ते में ही बंद हो गया। मुझे रिफंड चाहिए।",
   "{C} में मेरी बस की डिलीवरी 10 दिन लेट हुई और कोई फोन नहीं उठा रहा।",
   "वारंटी में {P} बदलवाना है, एक महीने में टूट गया।",
  ],
  "spam": [
   "{L} लाख तक का पर्सनल लोन {M} मिनट में पाएं। अभी क्लिक करें!",
   "बधाई हो! आपने फ्री iPhone जीता है। अभी क्लेम करें।",
   "इंस्टाग्राम फॉलोअर्स {F} बढ़ाएं, सस्ते रेट। अभी DM करें।",
   "घर बैठे रोज़ {E} कमाएं। WhatsApp करें।",
   "क्रिप्टो में निवेश करें और {D} दिन में पैसा डबल करें।",
   "गूगल पर पहले पेज की गारंटी के साथ SEO सर्विस पाएं।",
   "सारी साड़ियों पर एक के साथ एक फ्री। आज ही दुकान पर आएं।",
   "अपने बस बिज़नेस के लिए {L} लाख तक का लोन पाएं, कोई डॉक्यूमेंट नहीं। अभी क्लिक करें!",
   "अपने ट्रैवल बिज़नेस के लिए गारंटीड लीड्स पाएं। सीमित ऑफर, आज ही संपर्क करें।",
   "फ्री क्रेडिट कार्ड ऑफर, कोई वार्षिक शुल्क नहीं। अभी अप्लाई करें।",
  ],
 },
 "mr": {
  "price_query": [
   "{P} चा रेट काय आहे?",
   "{C} मध्ये {P} बसवायला किती खर्च येईल?",
   "कृपया {P} चे कोटेशन पाठवा. बजेट {B} आहे.",
   "{Q} साठी {P} ची किंमत सांगा.",
   "मला {P} ची किंमत जाणून घ्यायची आहे, बजेट साधारण {B}.",
   "तुम्ही {C} मध्ये {P} चे काम करता का? दर सांगा.",
   "{P} ची प्राईस लिस्ट पाठवा",
  ],
  "bulk_order": [
   "आम्हाला {Q} साठी {P} हवे आहे. घाऊक दर काय असेल?",
   "माझी ट्रॅव्हल कंपनी {C} मध्ये आहे, {Q} मध्ये {P} बसवायचे आहे. किती सूट मिळेल?",
   "{Q} चा बल्क ऑर्डर द्यायचा आहे, {P} हवे. बजेट {B}.",
   "आमच्याकडे {Q} आहेत, सर्वांमध्ये {P} बसवायचे आहे. कृपया कॉल करा.",
   "{C} मध्ये डिलिव्हरीसाठी {Q} मध्ये {P} हवे, 3 आठवड्यात.",
   "{Q} साठी {P} चा घाऊक दर सांगा, एकूण बजेट {B} आहे.",
  ],
  "complaint": [
   "तुम्ही बसवलेले {P} खराब झाले आहे. हे योग्य नाही.",
   "मी {P} चे पैसे दिले, पण काम अजून अपूर्ण आहे.",
   "{P} चे फिनिशिंग खूप वाईट आहे, ताबडतोब दुरुस्त करा.",
   "{P} एका आठवड्यातच बंद पडले. मला रिफंड हवा आहे.",
   "{C} मध्ये माझ्या बसची डिलिव्हरी 10 दिवस उशिरा झाली आणि कोणी फोन उचलत नाही.",
   "वॉरंटीमध्ये {P} बदलून हवे आहे, एका महिन्यात तुटले.",
  ],
  "spam": [
   "{L} लाखांपर्यंत पर्सनल लोन {M} मिनिटांत मिळवा. आत्ताच क्लिक करा!",
   "अभिनंदन! तुम्ही फ्री iPhone जिंकला आहे. आत्ताच क्लेम करा.",
   "इंस्टाग्राम फॉलोअर्स {F} वाढवा, स्वस्त दर. आत्ताच DM करा.",
   "घरबसल्या रोज {E} कमवा. WhatsApp करा.",
   "क्रिप्टोमध्ये गुंतवणूक करा आणि {D} दिवसांत पैसे दुप्पट करा.",
   "गुगलवर पहिल्या पानाची हमी असलेली SEO सेवा मिळवा.",
   "सर्व साड्यांवर एकावर एक फ्री. आजच दुकानात या.",
   "तुमच्या बस व्यवसायासाठी {L} लाखांपर्यंत कर्ज मिळवा, कागदपत्रे नाहीत. आत्ताच क्लिक करा!",
   "तुमच्या ट्रॅव्हल व्यवसायासाठी हमखास लीड्स मिळवा. मर्यादित ऑफर, आजच संपर्क साधा.",
   "फ्री क्रेडिट कार्ड ऑफर, वार्षिक शुल्क नाही. आत्ताच अर्ज करा.",
  ],
 },
 "mr_roman": {
  "price_query": [
   "{P} cha rate kay aahe?",
   "{C} madhe {P} basvayla kiti kharch yeil?",
   "please {P} che quotation pathva. budget {B} aahe",
   "{Q} sathi {P} chi kimmat sanga",
   "mala {P} chi kimmat janun ghyaychi aahe, budget saadharan {B}",
   "tumhi {C} madhe {P} che kaam karta ka? dar sanga",
   "{P} chi price list pathva",
  ],
  "bulk_order": [
   "amhala {Q} sathi {P} have aahe. ghau dar kay asel?",
   "majhi travels company {C} madhe aahe, {Q} madhe {P} basvaycha aahe. kiti sut milel?",
   "bulk order dyaycha aahe {Q} cha, {P} have. budget {B}",
   "amchyakade {Q} aahet, sarvamadhe {P} basvaycha aahe. please call kara",
   "{C} madhe delivery sathi {Q} madhe {P} have, 3 athavdyat",
   "{Q} sathi {P} cha ghau dar sanga, ekun budget {B} aahe",
  ],
  "complaint": [
   "tumhi basvlele {P} kharab zale aahe. he barobar nahi",
   "mi {P} che paise dile pan kaam ajun apurna aahe",
   "{P} che finishing khup vait aahe, tatkal durust kara",
   "{P} ek athavdyatach band padle. mala refund have",
   "{C} madhe majhya bus chi delivery 10 divas ushira zali ani koni phone uchalat nahi",
   "warranty madhe {P} badlun hava aahe, ek mahinyat tutle",
  ],
  "spam": [
   "{L} lakh paryant personal loan {M} minitat mila. aatach click kara!",
   "abhinandan! tumhi free iPhone jinkla aahe. aatach claim kara",
   "instagram followers {F} vadhva, swasta dar. DM kara",
   "gharbasalya roj {E} kamva. WhatsApp kara",
   "crypto madhe invest kara ani {D} divsat paise dupple kara",
   "google var pahilya panachi guarantee asleli SEO seva mila",
   "sarva sadyanvar ekavar ek free. aajach dukanat ya",
   "tumchya bus vyavsayasathi {L} lakh paryant loan mila, kagdapatre nahit. aatach click kara",
   "tumchya travel vyavsayasathi guaranteed leads mila. limited offer, aajach contact kara",
   "free credit card offer, annual fee nahi. aatach apply kara",
  ],
 },
}

URGENCY = {
    "en": ["Need it urgently.", "Please reply asap.", "Required by next week.", "Need this done this month."],
    "hinglish": ["urgent chahiye", "jaldi reply karo", "agle hafte tak chahiye", "aaj hi baat karni hai"],
    "hi": ["जल्दी चाहिए।", "बहुत ज़रूरी है।", "अगले हफ्ते तक चाहिए।", "आज ही बात करनी है।"],
    "mr": ["लवकर हवे आहे.", "खूप अर्जंट आहे.", "पुढच्या आठवड्यापर्यंत हवे.", "आजच बोलायचे आहे."],
    "mr_roman": ["lavkar have aahe", "khup urgent aahe", "pudhchya athavdyaparyant have", "aajach bolaycha aahe"],
}
GREET = {
    "en": ["Hi, ", "Hello sir, ", "Good morning, ", "Hey, "],
    "hinglish": ["hi ", "hello sir ", "bhai ", "namaste ", "sir "],
    "hi": ["नमस्ते, ", "हेलो सर, "],
    "mr": ["नमस्कार, ", "हॅलो सर, "],
    "mr_roman": ["namaskar, ", "hello sir, ", "dada, "],
}
CLOSER = {
    "en": ["Thanks.", "Thank you.", "Please call me."],
    "hinglish": ["thanks", "dhanyavad", "call karna"],
    "hi": ["धन्यवाद।", "कृपया कॉल करें।"],
    "mr": ["धन्यवाद.", "कृपया कॉल करा."],
    "mr_roman": ["dhanyavad", "call kara", "thanks"],
}

BASE = {"bulk_order": 50, "price_query": 30, "complaint": 20, "spam": 0}


def noisy(piece):
    """Light typo noise for Latin-script text (non-entity parts only)."""
    words = piece.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not idx:
        return piece
    i = random.choice(idx)
    w = words[i]
    j = random.randint(1, len(w) - 2)
    op = random.choice(["drop", "swap", "dup"])
    if op == "drop":
        w = w[:j] + w[j + 1:]
    elif op == "swap" and j + 1 < len(w):
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


SLOT = re.compile(r"\{([PQCBLMEFD])\}")
LABEL = {"P": "PRODUCT", "Q": "QTY", "C": "CITY", "B": "BUDGET"}


def build(lang, intent, template):
    """Fill one template; return (text, entities, urgency_flag)."""
    segs = []  # (text, label|None)
    pos = 0
    add_typos = lang in ROMAN and random.random() < 0.25
    fill = {"L": lambda: str(random.choice([1, 2, 3, 5, 10])), "M": lambda: str(random.choice([5, 10, 15, 30])),
            "E": lambda: str(random.choice([2000, 3000, 5000, 10000])),
            "F": lambda: random.choice(["1k", "5k", "10k", "50k"]), "D": lambda: str(random.choice([3, 5, 7, 10, 15]))}
    city_idx = random.randrange(len(CITIES["roman"]))
    for m in SLOT.finditer(template):
        lit = template[pos:m.start()]
        if lit:
            segs.append((noisy(lit) if add_typos else lit, None))
        k = m.group(1)
        if k == "P":
            segs.append((random.choice(PRODUCTS[lang]), "PRODUCT"))
        elif k == "Q":
            segs.append((make_qty(lang, intent), "QTY"))
        elif k == "C":
            segs.append((CITIES[CITY_KEY[lang]][city_idx], "CITY"))
        elif k == "B":
            segs.append((make_budget(lang, intent), "BUDGET"))
        else:
            segs.append((fill[k](), None))
        pos = m.end()
    tail = template[pos:]
    if tail:
        segs.append((noisy(tail) if add_typos else tail, None))

    urgency = 0
    if intent != "spam" and random.random() < 0.3:
        segs.append((" " + random.choice(URGENCY[lang]), None))
        urgency = 1
    if random.random() < (0.10 if intent == "spam" else 0.25):
        segs.append((" " + random.choice(CLOSER[lang]), None))
    if random.random() < (0.10 if intent == "spam" else 0.35):
        segs.insert(0, (random.choice(GREET[lang]), None))

    text = "".join(s for s, _ in segs)
    ents, cur = [], 0
    for s, lab in segs:
        if lab:
            ents.append({"start": cur, "end": cur + len(s), "label": lab, "text": s})
        cur += len(s)

    # casual formatting noise; both keep character offsets valid
    if lang in ROMAN and random.random() < 0.4:
        text = text.lower()
        for e in ents:
            e["text"] = e["text"].lower()
    if text and text[-1] in ".?!" and random.random() < 0.3:
        text = text[:-1]
    for e in ents:
        assert text[e["start"]:e["end"]] == e["text"], (text, e)
    return text, ents, urgency


def priority(intent, ents, urgency):
    if intent == "spam":
        return 0, "Ignore"
    labs = {e["label"] for e in ents}
    s = BASE[intent] + 15 * ("QTY" in labs) + 10 * ("BUDGET" in labs) + 5 * ("CITY" in labs) \
        + 5 * ("PRODUCT" in labs) + 15 * urgency
    s = min(s, 100)
    return s, ("High" if s >= 70 else "Medium" if s >= 40 else "Low")


TOKEN = re.compile(r"[^\s.,?!:;]+|[.,?!:;]")


def bio(text, ents):
    toks, tags = [], []
    for m in TOKEN.finditer(text):
        tag = "O"
        for e in ents:
            if m.start() >= e["start"] and m.end() <= e["end"]:
                tag = ("B-" if m.start() == e["start"] or not any(
                    t.startswith(("B-", "I-")) and tags and tags[-1] != "O" and tags[-1][2:] == e["label"]
                    for t in [tags[-1] if tags else "O"]) else "I-") + e["label"]
                break
        toks.append(m.group())
        tags.append(tag)
    return toks, tags


def main():
    rows, seen = [], set()
    rid = 1
    for lang in LANGS:
        for intent in INTENTS:
            tmpls = T[lang][intent]
            order = list(range(len(tmpls)))
            random.shuffle(order)
            n_test = max(1, round(len(order) * 0.15))
            n_val = max(1, round(len(order) * 0.15))
            split_of = {}
            for rank, ti in enumerate(order):
                split_of[ti] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
            per_t = PER_CELL // len(tmpls)
            extra = PER_CELL - per_t * len(tmpls)
            for ti, tmpl in enumerate(tmpls):
                want = per_t + (1 if ti < extra else 0)
                got, tries = 0, 0
                while got < want and tries < 3000:
                    tries += 1
                    text, ents, urg = build(lang, intent, tmpl)
                    if text in seen:
                        continue
                    seen.add(text)
                    score, label = priority(intent, ents, urg)
                    by = {e["label"]: e["text"] for e in ents}
                    toks, tags = bio(text, ents)
                    rows.append({
                        "id": f"LL{rid:05d}", "text": text, "language": lang, "intent": intent,
                        "product": by.get("PRODUCT", ""), "quantity": by.get("QTY", ""),
                        "city": by.get("CITY", ""), "budget": by.get("BUDGET", ""),
                        "urgency": urg, "priority_score": score, "priority_label": label,
                        "split": split_of[ti], "template_id": f"{lang}-{intent}-{ti}",
                        "source": "synthetic_template",
                        "_ents": ents, "_toks": toks, "_tags": tags,
                    })
                    rid += 1
                    got += 1
    random.shuffle(rows)

    cols = ["id", "text", "language", "intent", "product", "quantity", "city", "budget", "urgency",
            "priority_score", "priority_label", "split", "template_id", "source"]
    with open(OUT / "leadlens_dataset.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    with open(OUT / "leadlens_ner.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "id": r["id"], "text": r["text"], "language": r["language"], "intent": r["intent"],
                "split": r["split"], "priority_score": r["priority_score"], "urgency": r["urgency"],
                "entities": r["_ents"], "tokens": r["_toks"], "ner_tags": r["_tags"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
