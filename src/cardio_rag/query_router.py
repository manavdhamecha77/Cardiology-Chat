import re

INTENT_TERMS = {
    "definition": ("what is", "define", "meaning", "definition"),
    "risk_factor": ("risk factor", "increase risk", "cause", "causes", "prevention"),
    "symptom_condition": ("symptom", "sign", "condition", "disease", "heart failure", "arrhythmia"),
    "test": ("test", "diagnos", "ecg", "ekg", "echocardiogram", "measure"),
    "treatment": ("treatment", "treat", "therapy", "medication", "manage"),
    "comparison": ("compare", "difference", "versus", " vs ", "compared"),
}


def classify_intent(question: str) -> str:
    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', question.lower())} "
    for intent, phrases in INTENT_TERMS.items():
        if any(phrase in normalized for phrase in phrases):
            return intent
    return "other"


def adapt_query(question: str, intent: str) -> str:
    hints = {
        "definition": "definition overview",
        "risk_factor": "risk factors prevention",
        "symptom_condition": "condition symptoms signs",
        "test": "diagnostic test indications",
        "treatment": "management treatment options",
        "comparison": "comparison differences",
        "other": "cardiology educational information",
    }
    return f"{question} {hints[intent]}"
