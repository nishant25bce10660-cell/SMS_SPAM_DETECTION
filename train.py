import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_PATH = os.path.join("data", "spam.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding="latin-1")

if {"v1", "v2"}.issubset(df.columns):
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "text"]
else:
    raise ValueError("CSV does not have expected columns 'v1' and 'v2'.")

df = df.dropna(subset=["text", "label"])
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
if df["label_num"].isnull().any():
    bad = df[df["label_num"].isnull()]["label"].unique()
    raise ValueError(f"Found unexpected label values: {bad}")


print("First few rows of data:")
print(df.head())


df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

df = df.dropna(subset=["text"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label_num"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"]
)

print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2), 
    min_df=2            
)


X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_proba = model.predict_proba(X_test_tfidf)[:, 1]  
threshold = 0.4  

import numpy as np
y_pred = (y_proba >= threshold).astype(int)

print("\nEvaluation on test set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Vectorizer saved to: {VECTORIZER_PATH}")
print("\nTraining complete.")


X_test_series = X_test.reset_index(drop=True)
y_test_series = y_test.reset_index(drop=True)

print("\nSome FALSE POSITIVES (ham predicted as spam):\n")
fp_count = 0
for text, true_label, pred_label in zip(X_test_series, y_test_series, y_pred):
    if true_label == 0 and pred_label == 1:  
        print(f"[HAM -> SPAM] {text[:120]}...")
        fp_count += 1
        if fp_count >= 5:
            break

print("\nSome FALSE NEGATIVES (spam predicted as ham):\n")
fn_count = 0
for text, true_label, pred_label in zip(X_test_series, y_test_series, y_pred):
    if true_label == 1 and pred_label == 0: 
        print(f"[SPAM -> HAM] {text[:120]}...")
        fn_count += 1
        if fn_count >= 5:
            break
