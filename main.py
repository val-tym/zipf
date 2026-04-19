import re
import requests
import random
import stanza
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import os

# -------------------------------
# НАЛАШТУВАННЯ
# -------------------------------
TARGET_TOKENS = 80000
TOP_N = 20
LANGUAGES = ["en", "fr", "de"]

# -------------------------------
# 0. Папка результатів
# -------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"results_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUTPUT_DIR, "report.txt")

def write_report(text):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# -------------------------------
# 1. Gutenberg API
# -------------------------------
def get_books(lang):
    url = f"https://gutendex.com/books?languages={lang}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["results"]

def get_random_book_url(lang):
    books = get_books(lang)
    while True:
        book = random.choice(books)
        for k, v in book["formats"].items():
            if "text/plain" in k:
                return v

# -------------------------------
# 2. Завантаження
# -------------------------------
def download_text(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.text

# -------------------------------
# 3. Обрізка Gutenberg
# -------------------------------
def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF.*?\*\*\*", text)
    if start and end:
        return text[start.end():end.start()]
    return text

# -------------------------------
# 4. Очистка
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"'s\b", "", text)  # remove possessive 's
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha()]  # ❗ прибираємо числа
    return tokens

# -------------------------------
# 5. Накопичення токенів
# -------------------------------
def collect_tokens(lang):
    tokens = []
    while len(tokens) < TARGET_TOKENS:
        url = get_random_book_url(lang)
        print(f"[{lang}] {url}")
        try:
            text = download_text(url)
            text = strip_gutenberg(text)
            tokens.extend(preprocess(text))
        except:
            continue
    return tokens[:TARGET_TOKENS]

# -------------------------------
# 6. Лематизація
# -------------------------------
def lemmatize(tokens, lang):
    stanza.download(lang)
    nlp = stanza.Pipeline(lang=lang, processors="tokenize,lemma", use_gpu=True)
    doc = nlp(" ".join(tokens))
    return [w.lemma for s in doc.sentences for w in s.words if w.lemma.isalpha()]

# -------------------------------
# 7. Zipf + збереження
# -------------------------------
def zipf_analysis(tokens, title, filename_prefix):
    freq = Counter(tokens)
    sorted_freq = sorted(freq.values(), reverse=True)

    ranks = np.arange(1, len(sorted_freq) + 1)
    freqs = np.array(sorted_freq)

    log_r = np.log(ranks)
    log_f = np.log(freqs)

    slope, intercept = np.polyfit(log_r, log_f, 1)
    s = -slope

    # графік
    plt.figure()
    plt.scatter(log_r, log_f, s=5)
    plt.plot(log_r, slope * log_r + intercept)
    plt.xlabel("log(rank)")
    plt.ylabel("log(freq)")
    plt.title(title)

    plot_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    plt.savefig(plot_path)
    plt.close()

    return s

# -------------------------------
# 8. ТОП слова
# -------------------------------
def get_top(tokens):
    return Counter(tokens).most_common(TOP_N)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    write_report(f"Experiment started: {timestamp}")
    write_report(f"Target tokens: {TARGET_TOKENS}\n")

    for lang in LANGUAGES:
        print(f"\n=== {lang} ===")
        write_report(f"\n=== LANGUAGE: {lang} ===")

        tokens = collect_tokens(lang)

        # RAW
        s_raw = zipf_analysis(tokens, f"{lang} raw", f"{lang}_raw")
        top_raw = get_top(tokens)

        write_report(f"\nRAW s ≈ {s_raw:.4f}")
        write_report("Top words (raw):")
        for w, c in top_raw:
            write_report(f"{w:15} {c}")

        # LEMMA
        lemmas = lemmatize(tokens, lang)

        s_lemma = zipf_analysis(lemmas, f"{lang} lemma", f"{lang}_lemma")
        top_lemma = get_top(lemmas)

        write_report(f"\nLEMMA s ≈ {s_lemma:.4f}")
        write_report("Top words (lemma):")
        for w, c in top_lemma:
            write_report(f"{w:15} {c}")

    print(f"\nResults saved in: {OUTPUT_DIR}")