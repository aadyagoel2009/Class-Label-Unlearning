import os
import re
import math
from pathlib import Path
import pickle
from collections import Counter, defaultdict

from openai import OpenAI
from sklearn.datasets import fetch_20newsgroups

# ──────────────────────────────────────────────────────────────────────────────
# DeepSeek client setup
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path(__file__).resolve().parent
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# ──────────────────────────────────────────────────────────────────────────────
# Config & cache paths
# ──────────────────────────────────────────────────────────────────────────────
DOC_KWS_K        = 10
CLASS_CACHE      = BASE_DIR / "testing_class.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Get & cache class_importances
# ──────────────────────────────────────────────────────────────────────────────
def build_and_cache_top_words(docs):
    # If we’ve already run this once, just load the cache
    if CLASS_CACHE.exists():
        return pickle.load(open(CLASS_CACHE, "rb"))

    doc_kws = {}
    for idx, doc in enumerate(docs):
        print("start")
        instruction = (
            f"Identify the {DOC_KWS_K} most distinctive, generalizable keywords or short phrases "
            "that capture the core themes of this document in a way that would help categorize "
            "*other* similar texts. Focus on topic or concept terms, and exclude overly "
            "document-specific proper names. Output exactly one `term:score` per line, where "
            "`score` is a decimal between 0.0 and 1.0. No extra text."
        )
        messages = [
            {"role": "system", "content": "Output only lines of `term:score`."},
            {"role": "user",   "content": doc[:8000] + "\n\n" + instruction}
        ]

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

        terms = []
        for line in resp.choices[0].message.content.splitlines():
            if ":" not in line:
                continue
            term, scr = line.split(":", 1)
            try:
                terms.append((term.strip(), float(scr)))
            except ValueError:
                continue

        # Store the **full document** and its keywords+scores under its index
        doc_kws[idx] = {
            "doc":  doc,
            "kws":  terms
        }
        print("stop")

    # Write the cache out
    with open(CLASS_CACHE, "wb") as f:
        pickle.dump(doc_kws, f)

    return doc_kws

# ──────────────────────────────────────────────────────────────────────────────
# Main: Build & save caches
# ──────────────────────────────────────────────────────────────────────────────
def main():
    data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
    docs, labels = data.data, data.target
    split = int(0.8 * len(docs))
    train_docs, test_docs = docs[:split], docs[split:]
    train_lbls, test_lbls = labels[:split], labels[split:]

    doc2 = train_docs[:2]

    top_words = build_and_cache_top_words(doc2)

    for idx, info in top_words.items():
        print(f"Document #{idx}:")
        print(info["doc"][:200].replace("\n"," "), "…")
        print("Top keywords + scores:", info["kws"])
        print()

if __name__ == "__main__":
    main()

