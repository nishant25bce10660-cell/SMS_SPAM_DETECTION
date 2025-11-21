# predict.py
import os
import sys
import joblib
import math
import re

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# A small list of strong spam indicators / profanity that should bias prediction toward SPAM.
# You can add/remove words as you see fit for your dataset / policy.
STRONG_SPAM_WORDS = {
    "free", "prize", "winner", "win", "cash", "claim", "lottery", "urgent",
    "congrat", "click", "offer", "buy", "discount", "credit", "limited",
    "cheap", "sex", "porn", "fuck", "f***", "loan", "winner", "selected"
}

def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("❌ Model or vectorizer not found. Please run train.py first.")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Loaded vectorizer: {VECTORIZER_PATH}")
    return model, vectorizer

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def get_spam_probability(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return float(proba[1])  # prob of class 1 = spam
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        return float(sigmoid(score))
    else:
        return None

def simple_preprocess(text: str) -> str:
    """
    Preprocess text in the same spirit as training:
    - Lowercase
    - Replace URLs/emails with token
    - Replace non-word characters with spaces (keeps words)
    - Collapse multiple spaces
    This helps when users type casually without punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # replace urls & emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # keep alphanumeric characters and apostrophes, replace others with space
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    # collapse repeated letters optionally? (not doing that automatically)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_strong_spam_word(text: str) -> bool:
    # Tokenize simply by split after preprocessing
    tokens = text.split()
    for t in tokens:
        # check if token starts with any strong spam word (handles 'congratulations' -> 'congrat')
        for sw in STRONG_SPAM_WORDS:
            if t.startswith(sw):
                return True
    return False

def classify_with_boost(model, vectorizer, raw_message: str, boost_threshold=0.80):
    """
    Returns (pred_label, spam_prob, reason)
    - pred_label: 1 => SPAM, 0 => HAM
    - spam_prob: estimated probability (0..1) from model or approximated
    - reason: text explaining if boosted due to strong-word rule
    """
    pre = simple_preprocess(raw_message)
    X = vectorizer.transform([pre])
    spam_prob = get_spam_probability(model, X)
    pred = int(model.predict(X)[0])

    reason = "model_only"

    # If any strong spam word is present, boost the spam decision:
    if contains_strong_spam_word(pre):
        # If model already very confident spam, keep it.
        # Otherwise, set spam_prob to at least boost_threshold and pred = spam.
        if spam_prob is None or spam_prob < boost_threshold:
            spam_prob = max(spam_prob or 0.0, boost_threshold)
        pred = 1
        reason = "boosted_by_strong_word"

    return pred, spam_prob, reason

def pretty_output(raw_message: str, pred: int, spam_prob, reason: str):
    emoji = "🔴" if pred == 1 else "🟢"
    label = "SPAM" if pred == 1 else "NOT SPAM (HAM)"
    lines = [f"{emoji} Prediction: {label}"]
    if spam_prob is not None:
        lines.append(f"(Estimated spam probability: {spam_prob:.3f})")
    lines.append(f"(Decision info: {reason})")
    return "\n".join(lines)

def run_cli_mode(model, vectorizer, message: str):
    print("\nInput message:")
    print(message)
    pred, spam_prob, reason = classify_with_boost(model, vectorizer, message)
    print("\nResult:")
    print(pretty_output(message, pred, spam_prob, reason))

def run_interactive_mode(model, vectorizer):
    print("\n=== SMS Spam Detector (interactive) ===")
    print("Type an SMS message and press Enter to classify it.")
    print("Press Enter on an empty line to exit.\n")
    # Loop and accept input, catching EOF / KeyboardInterrupt
    while True:
        try:
            message = input("Enter SMS text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye! 👋")
            break

        if message is None:
            # should not happen, but safe guard
            continue
        message = message.strip()
        if message == "":
            print("\nExiting. Goodbye! 👋")
            break

        pred, spam_prob, reason = classify_with_boost(model, vectorizer, message)
        print("\n" + pretty_output(message, pred, spam_prob, reason) + "\n")

if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()

    # If a message is provided as CLI args, use that (good for screenshots)
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:]).strip()
        if message == "":
            print("Empty message provided. Exiting.")
            sys.exit(1)
        run_cli_mode(model, vectorizer, message)
    else:
        # If stdin is not a tty (no interactive), print usage instead of exiting silently
        if not sys.stdin.isatty():
            print("No interactive terminal detected. To use, run:")
            print('  python predict.py "Your SMS message here"')
            sys.exit(1)
        run_interactive_mode(model, vectorizer)


