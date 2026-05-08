from datasets import load_dataset
import pandas as pd
import random
import os


# =========================================
# CONFIG
# =========================================

HUMAN_SAMPLES = 3000

GPT_FILE = "gpt.csv"
GEMINI_FILE = "gemini.csv"
CLAUDE_FILE = "claude.csv"

OUTPUT_FILE = "final_dataset.csv"


# =========================================
# LOAD + GENERATE HUMAN DATA
# =========================================

print("Loading human datasets...")


# ---------- Tweets ----------
tweet_dataset = load_dataset(
    "tweet_eval",
    "sentiment",
    split=f"train[:1500]"
)

tweet_texts = [
    item["text"]
    for item in tweet_dataset
]


# ---------- AG News ----------
news_dataset = load_dataset(
    "ag_news",
    split=f"train[:1500]"
)

news_texts = [
    item["text"]
    for item in news_dataset
]


# ---------- Combine ----------
human_texts = tweet_texts + news_texts


# Clean
human_texts = [
    text.strip()
    for text in human_texts
    if len(text.split()) >= 4
]


# Human dataframe
human_df = pd.DataFrame({
    "text": human_texts,
    "label": 0,
    "source": "human"
})

print(f"Collected {len(human_df)} human samples")


# =========================================
# LOAD AI CSV FILES
# =========================================

print("\nLoading AI-generated datasets...")


def load_ai_csv(file_path, source_name):

    if not os.path.exists(file_path):
        print(f"WARNING: {file_path} not found")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.read_csv(file_path)

    # Keep only required column
    if "text" not in df.columns:
        raise ValueError(f"'text' column missing in {file_path}")

    df = df[["text"]].copy()

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()

    # Remove short rows
    df = df[
        df["text"].apply(lambda x: len(x.split()) >= 4)
    ]

    # Add labels
    df["label"] = 1
    df["source"] = source_name

    print(f"Loaded {len(df)} samples from {file_path}")

    return df


# ---------- Load GPT ----------
gpt_df = load_ai_csv(GPT_FILE, "gpt")

# ---------- Load Gemini ----------
gemini_df = load_ai_csv(GEMINI_FILE, "gemini")

# ---------- Load Claude ----------
claude_df = load_ai_csv(CLAUDE_FILE, "claude")


# =========================================
# COMBINE ALL DATA
# =========================================

print("\nCombining datasets...")


final_df = pd.concat(
    [
        human_df,
        gpt_df,
        gemini_df,
        claude_df
    ],
    ignore_index=True
)


# =========================================
# CLEAN DATASET
# =========================================

print("Cleaning dataset...")


# Remove NaN
final_df.dropna(inplace=True)

# Remove duplicates
final_df.drop_duplicates(
    subset=["text"],
    inplace=True
)

# Shuffle dataset
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================================
# SAVE DATASET
# =========================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset creation completed!")

print(f"\nSaved as: {OUTPUT_FILE}")

print("\nClass Distribution:")
print(final_df["label"].value_counts())

print("\nSource Distribution:")
print(final_df["source"].value_counts())

print("\nSample Rows:")
print(final_df.head())