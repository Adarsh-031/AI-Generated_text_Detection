import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# Text Cleaning Function
# =========================

def clean_text(text):
    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"\W", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# Load Dataset
# =========================

print("Loading dataset...")

df = pd.read_csv("../data/Data1.csv")

# Select required columns
df = df[["text", "label"]]

# Rename columns
df.columns = ["text", "label"]

print("Dataset loaded successfully!")
print(df.head())


# =========================
# Preprocess Text
# =========================

print("\nCleaning text...")

df["text"] = df["text"].apply(clean_text)

print("Text preprocessing completed!")


# =========================
# Train-Test Split
# =========================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-test split completed!")


# =========================
# TF-IDF Vectorization
# =========================

print("\nVectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Vectorization completed!")


# =========================
# Train Model
# =========================

print("\nTraining model...")

model = LinearSVC()

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# =========================
# Evaluate Model
# =========================

print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# =========================
# Save Model & Vectorizer
# =========================

print("\nSaving model files...")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")

print("\nTraining pipeline completed successfully!")