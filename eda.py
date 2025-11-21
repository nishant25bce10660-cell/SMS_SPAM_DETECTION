import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "spam.csv")

def run_eda(data_path=DATA_PATH, save_plots=False, out_dir="plots"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path, encoding="latin-1", low_memory=False)

    
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    else:
        raise ValueError("Expected columns 'v1' and 'v2' in spam.csv")

    df["text"] = df["text"].fillna("")
    label_counts = df["label"].value_counts()
    print(label_counts)

    plt.figure()
    label_counts.plot(kind="bar")
    plt.title("Count of HAM vs SPAM messages")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_plots:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "label_counts.png"))
    else:
        plt.show()

    df["length"] = df["text"].str.len()
    plt.figure()
    df[df["label"] == "ham"]["length"].plot(kind="hist", alpha=0.5, label="ham")
    df[df["label"] == "spam"]["length"].plot(kind="hist", alpha=0.5, label="spam")
    plt.legend()
    plt.title("Message length distribution (ham vs spam)")
    plt.xlabel("Message length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(out_dir, "length_dist.png"))
    else:
        plt.show()

if __name__ == "__main__":
    run_eda()