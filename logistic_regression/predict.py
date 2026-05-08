import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_text(text):
    cleaned_text = clean_text(text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "AI Generated"

    return "Human Written"


# User Input
text = input("Enter text: ")

print("\nPrediction:", predict_text(text))