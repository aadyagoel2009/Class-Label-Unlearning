import os
import re
import math
import pickle
from pathlib import Path
from collections import Counter

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
DOC_KWS_CACHE     = BASE_DIR / "top_words.pkl"       # your per‐doc training kws + scores
TEST_SCORES_CACHE = BASE_DIR / "test_scores.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Load your training-doc kws cache and extract the global keyword set
# ──────────────────────────────────────────────────────────────────────────────
with open(DOC_KWS_CACHE, "rb") as f:
    doc_kws: dict[int, dict] = pickle.load(f)
# doc_kws[i] == {"doc": <text>, "kws": [(term,score),…]}

# build a global set of keywords from your training docs
raw_terms = { term
    for info in doc_kws.values()
    for term,_ in info["kws"]
}

global_terms = { t.replace("_", " ") for t in raw_terms }

# ──────────────────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────────────────
def char_entropy(word: str) -> float:
    cnt = Counter(word)
    total = len(word)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in cnt.values())

def clean_data(text: str) -> list[str]:
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and len(w)>=2]
    return [w for w in words if 1.0 <= char_entropy(w) <= 4.0]

# ──────────────────────────────────────────────────────────────────────────────
# Score & cache test-doc term relevances
# ──────────────────────────────────────────────────────────────────────────────
def score_and_cache_test_docs(test_docs: list[str]):
    if TEST_SCORES_CACHE.exists():
        return pickle.load(open(TEST_SCORES_CACHE, "rb"))

    test_scores: dict[int, dict] = {}
    for idx, doc in enumerate(test_docs):
        print(idx)
        # pick only the global terms present in this doc
        candidates   = [
            t for t in global_terms if t in doc
        ]

        if not candidates:
            test_scores[idx] = {"doc": doc, "kws": []}
            continue

        # ask DeepSeek to score them
        term_list = "\n".join(candidates)
        instruction = (
            "Score each of these terms for how well they describe this document "
            "on a 0.0–1.0 scale. Output one `term:score` per line, no extra text."
        )
        messages = [
            {"role":"system","content":"Output only lines of `term:score`."},
            {"role":"user","content": doc[:8000] + "\n\n" + instruction + "\n\nTerms:\n" + term_list}
        ]

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

        kws: list[tuple[str,float]] = []
        for line in resp.choices[0].message.content.splitlines():
            if ":" not in line:
                continue
            term, scr = line.split(":", 1)
            try:
                kws.append((term.strip(), float(scr)))
            except ValueError:
                continue

        test_scores[idx] = {"doc": doc, "kws": kws}

    # cache it
    with open(TEST_SCORES_CACHE, "wb") as f:
        pickle.dump(test_scores, f)

    return test_scores

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    data      = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
    docs, _   = data.data, data.target
    split     = int(0.8 * len(docs))
    test_docs = docs[split:]

    test_scores = score_and_cache_test_docs(test_docs)
    print(f"Saved scores for {len(test_scores)} test docs to {TEST_SCORES_CACHE}")

    # Example inspect:
    for idx, info in test_scores.items():
        print(f"Doc #{idx} has {len(info['kws'])} keywords:")
        for term, score in info["kws"]:
            print(f"  {term}: {score:.3f}")
        print()

if __name__ == "__main__":
    main()